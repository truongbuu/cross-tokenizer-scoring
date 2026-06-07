from dataclasses import dataclass
from typing import Optional, Tuple, List

import logging
import copy
import torch
import numpy as np

from transformers import Qwen2Tokenizer
from base_model import BaseModel


@dataclass
class SamplerState:
    P_tT: torch.Tensor
    cvrs_logprobs: torch.Tensor
    covers: List[str]
    cover_encs: List[List[int]]
    prev_context_enc: List[int]
    div_enc: List[int]
    old_cache: Optional[Tuple]
    old_cache_enc: Optional[List[int]]
    new_cache: Optional[Tuple]
    new_cache_enc: Optional[List[int]]


def extract_sampler_state(sampler_state: SamplerState):
    """
    Unpack and return the components of a SamplerState.
    """
    P_tT = sampler_state.P_tT
    cvrs_logprobs = sampler_state.cvrs_logprobs
    covers = sampler_state.covers
    cover_encs = sampler_state.cover_encs
    prev_context_enc = sampler_state.prev_context_enc
    div_enc = sampler_state.div_enc

    assert prev_context_enc[:len(div_enc)] == div_enc

    return (
        P_tT,
        cvrs_logprobs,
        covers,
        cover_encs,
        prev_context_enc,
        div_enc,
    )


def log2prob_rescale(cvrs_logprobs: torch.Tensor):
    """
    Rescale log-probabilities so max value becomes 0.
    """
    max_log = torch.max(cvrs_logprobs)
    cvrs_logprobs -= max_log
    cvrs_scaled_probs = torch.exp(cvrs_logprobs)
    return cvrs_scaled_probs, cvrs_logprobs


def whitespace_split(white_space: str, raw_str: str):
    """
    Split `raw_str` into prefix and remainder after the last whitespace following a word.
    """
    break_point = -1
    for i in range(1, len(raw_str)):
        if raw_str[i] == white_space and raw_str[i - 1] != white_space:
            break_point = i

    assert break_point > 0, "Invalid string: no valid whitespace split found."

    cond_str = raw_str[:break_point]
    query_str = raw_str[break_point:]
    return cond_str, query_str


def insert_sequence(trie: dict, seq: List[int]):
    """
    Insert a sequence of tokens into a trie.
    """
    node = trie
    for token in seq:
        if token not in node:
            node[token] = {}
        node = node[token]
    node["$"] = True  # Marks end of sequence


def find_all_from_prefix(trie: dict, prefix: List[int]):
    """
    Find all sequences in the trie that start with the given prefix.
    """
    node = trie
    for token in prefix:
        if token not in node:
            return []
        node = node[token]

    results = []

    def dfs(current_node, path):
        if "$" in current_node:
            results.append(prefix + path)
        for token, child in current_node.items():
            if token == "$":
                continue
            dfs(child, path + [token])

    dfs(node, [])
    return results


class SubVocabPredLLM(BaseModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: str = "cpu",
        sub_vocab_merge: str = "Qwen2/first_vocab_10000_subset.json",
        sub_merges_file: str = "Qwen2/merge_10000.txt",
        vocab_file="Qwen2/vocab.json",
        merge_file="Qwen2/merges.txt"
    ):
        super().__init__(pretrained_model_name_or_path, device, vocab_file=vocab_file, merge_file=merge_file)

        self.sub_tokenizer = Qwen2Tokenizer(
            vocab_file=sub_vocab_merge,
            merges_file=sub_merges_file,
            fast_mode=False
        )
        self.device = device
        
        self.white_space = 'Ġ'

        # Build trie of subtoken sequences
        self.create_trie()

        # Create sparse prefix matrix
        self.create_prefix_mat()
        
        # Create map subtoken --> big token
        self.create_sub2bigtoken()
    
    def create_trie(self):
        self.trie = {}
        for i in range(self.orig_tokenizer.vocab_size):
            token = self.orig_tokenizer.convert_ids_to_tokens(i)
            sub_tokens = self.sub_tokenizer.convert_tokens_to_ids(self.sub_tokenizer.bpe(token).split(" "))
            insert_sequence(self.trie, sub_tokens)

        # Create sparse prefix matrix
        prefix_mat = torch.zeros(
            self.sub_tokenizer.vocab_size,
            self.orig_tokenizer.vocab_size,
        )
        
    def create_prefix_mat(self):
        prefix_mat = torch.zeros(
            self.sub_tokenizer.vocab_size,
            self.orig_tokenizer.vocab_size,
        )

        for i in range(self.orig_tokenizer.vocab_size):
            token = self.orig_tokenizer.convert_ids_to_tokens(i)
            sub_tokens = self.sub_tokenizer.convert_tokens_to_ids(self.sub_tokenizer.bpe(token).split(" "))
            prefix_mat[sub_tokens[0], i] = 1

        indices = torch.nonzero(prefix_mat, as_tuple=False).t()
        values = prefix_mat[indices[0], indices[1]]
        self.sparse_prefix_mat = torch.sparse_coo_tensor(
            indices, values, prefix_mat.shape
        ).to(self.device)
    
    def create_sub2bigtoken(self):
        self.map_subtoken_bigtoken = {}
        for i in range(self.orig_tokenizer.vocab_size):
            token = self.orig_tokenizer.convert_ids_to_tokens(i)
            sub_tokens = self.sub_tokenizer.convert_tokens_to_ids(self.sub_tokenizer.bpe(token).split(" "))
            first_subtoken = sub_tokens[0]

            if first_subtoken not in self.map_subtoken_bigtoken:
                self.map_subtoken_bigtoken[first_subtoken] = []

            self.map_subtoken_bigtoken[first_subtoken].append(i)

    def check_valid_enc(self, list_encs: List[List[int]], tokenizer):
        """
        Keep only encodings whose decode-then-encode roundtrip is unchanged.
        """
        output_encs = []
        for enc in list_encs:
            tokens = tokenizer.convert_ids_to_tokens(enc)
            true_enc = tokenizer.convert_tokens_to_ids(tokenizer.bpe("".join(tokens)).split(" "))
            if true_enc == enc:
                output_encs.append(enc)
        return output_encs

    def cover_token_likelihood(self, t_m1: List[int], t_n1: List[int]):
        """
        Compute likelihood of token sequences under the original tokenizer.
        """
        assert t_m1[:len(t_n1)] == t_n1
        logprobs, _ = self.logprobs(input_ids=t_m1, use_cache=False)
        last_logprob = logprobs[0, -1, :]
        x = list(range(len(t_n1) - 1, len(t_m1) - 1))
        y = t_m1[len(t_n1):]
        logp_mn = logprobs[0, x, y].sum()
        likelihood_per_encs = last_logprob + logp_mn
        return likelihood_per_encs
    
    def enc2symbols(self, encodings, tokenizer):
        # Convert encodings to symbols format.
        # Note that symbols is different from string. Like the white space thing.
        symbols = "".join(tokenizer.convert_ids_to_tokens(encodings))
        return symbols
    
    def symbols2enc(self, symbols, tokenizer):
        # Convert list of symbols into list of token ids
        encs = tokenizer.convert_tokens_to_ids(tokenizer.bpe(symbols).split(" "))
        return encs
    

    def extract_cover_encodings(self, encodings: List[int], debug=False):
        """
        For given subtoken encodings, extract possible cover encodings
        in the original vocabulary.
        """
        inp_sym = self.enc2symbols(encodings, self.sub_tokenizer) #Recheck if necessary
        cond_syms, query_syms = whitespace_split(self.white_space, inp_sym)

        def supertoken_V(encs_V_star):
            supertokens = find_all_from_prefix(self.trie, encs_V_star)
            supertokens_V = []
            for supertoken in supertokens:
                t = self.symbols2enc(
                    self.enc2symbols(supertoken, self.sub_tokenizer), self.orig_tokenizer
                )
                assert len(t) == 1
                supertokens_V.append(t)
            return supertokens_V

        cond_str_enc_V = self.symbols2enc(cond_syms, self.orig_tokenizer)  
        query_str_enc_V_star = self.symbols2enc(query_syms, self.sub_tokenizer)

        cover_Vencs = []
        log_prob_cover_Vencs = []
        cover_Vsub = []

        for i in range(len(query_str_enc_V_star)):
            left_encs_V_star = query_str_enc_V_star[:i]
            right_encs_V_star = query_str_enc_V_star[i:]

            if len(right_encs_V_star) == 0:
                continue
            #print (left_encs_V_star)
            if len(left_encs_V_star) == 0:
                left_encs_V = []
            else:
                left_encs_V = self.symbols2enc(
                    self.enc2symbols(left_encs_V_star, self.sub_tokenizer),
                    self.orig_tokenizer
                ) 
            
            right_encs_V = supertoken_V(right_encs_V_star)
            proposal_encs_V = [
                left_encs_V + enc_V for enc_V in right_encs_V
            ]
            refined_encs_V = self.check_valid_enc(
                proposal_encs_V, self.orig_tokenizer
            )

            t_m1 = cond_str_enc_V + left_encs_V
            t_n1 = cond_str_enc_V
            logp_nexttoken = self.cover_token_likelihood(t_m1, t_n1)

            for enc in refined_encs_V:
                cover_Vencs.append(enc)
                to_add_cover_Vsub = self.symbols2enc(
                    self.enc2symbols(enc, self.orig_tokenizer), self.sub_tokenizer
                )
                cover_Vsub.append(to_add_cover_Vsub)
                log_prob_cover_Vencs.append(logp_nexttoken[enc[-1]])

        return (
            cover_Vsub,
            cover_Vencs,
            torch.stack(log_prob_cover_Vencs).to(self.device),
            cond_str_enc_V
        )
    
                                
    def prob_next_subtoken(
        self,
        raw_subencoding: List[int],
        fast_mode=False,
        sampler_state: Optional[SamplerState] = None,
        debug=False,
    ):
        inp_sym = self.enc2symbols(raw_subencoding, self.sub_tokenizer) 
        cond_sym, query_sym = whitespace_split(self.white_space, inp_sym)
        last_subtoken = raw_subencoding[-1]
        ctx_V_enc = self.symbols2enc(inp_sym, self.orig_tokenizer) 

        if sampler_state is None:
            raw_cover_Vsub, cover_Vencs, log_prob_cover_Vencs, cond_str_enc_V = (
                self.extract_cover_encodings(raw_subencoding)
            )
            assert ctx_V_enc[len(cond_str_enc_V):] in cover_Vencs
            cover_Vsub = [
                cover[len(self.symbols2enc(query_sym, self.sub_tokenizer)):]
                for cover in raw_cover_Vsub
            ]
        else:
            (
            log_prob_cover_Vencs,
            cover_Vsub,
            cover_Vencs,
            cond_str_enc_V
            ) = self.update_sampler_state(sampler_state, last_subtoken, ctx_V_enc)
            

        log_pt_T, cache = self._run_kv(ctx_V_enc, sampler_state, cond_str_enc_V)
        #self.logprobs(input_ids=ctx_V_enc, use_cache=False)
        log_pt_T = log_pt_T[0, -1, :self.orig_tokenizer.vocab_size]
        cvrs_scaled_probs, log_prob_cover_Vencs = log2prob_rescale(
            log_prob_cover_Vencs
        )
        
        # step 2:
        log_pT_V = log_prob_cover_Vencs[
            cover_Vencs.index(ctx_V_enc[len(cond_str_enc_V):])
        ]
        ptT_V = torch.exp(log_pT_V + log_pt_T)
        p_atom_T = torch.sparse.mm(
            self.sparse_prefix_mat, ptT_V.view(-1, 1)
        )

        mat_sum = torch.zeros(
            len(p_atom_T), len(cvrs_scaled_probs), dtype=torch.float32
        )
        temp_list = []
        for k, sub in enumerate(cover_Vsub):
            if not sub:
                continue
            temp_list.append([sub[0], k])
        if temp_list:
            temp_array = np.array(temp_list)
            mat_sum[temp_array[:, 0], temp_array[:, 1]] = 1.0

        P_unorm = (
            torch.matmul(mat_sum.to(self.device), cvrs_scaled_probs)
            + p_atom_T[:, 0]
        )
        predictions = P_unorm / P_unorm.sum()
        
        # Update the kv cache.
        if sampler_state is None:
            old_cache_enc = cond_str_enc_V
            _, old_cache = self.logprobs(input_ids=ctx_V_enc[: len(old_cache_enc)],\
                                         past_key_values=None, use_cache=True)
            #ctx_enc[: len(old_cache_enc)], fast_mode, cache=None, use_cache=True
            #)
        else:
            old_cache = sampler_state.old_cache
            old_cache_enc = sampler_state.old_cache_enc
        
        if self.sub_tokenizer.encode(self.sub_tokenizer.decode(raw_subencoding)) == raw_subencoding:
            #see if string is a valid utf8
            new_cache = cache
            new_cache_enc = ctx_V_enc
        else:
            if sampler_state is not None:
                new_cache = sampler_state.new_cache
                new_cache_enc = sampler_state.new_cache_enc
            else:
                new_cache = None
                new_cache_enc = None

        sampler_state = SamplerState(
            P_tT=ptT_V,
            cvrs_logprobs=log_prob_cover_Vencs,
            covers=cover_Vsub,
            cover_encs=cover_Vencs,
            prev_context_enc=ctx_V_enc,
            div_enc=cond_str_enc_V,
            old_cache=old_cache,
            old_cache_enc=old_cache_enc,
            new_cache=new_cache,
            new_cache_enc=new_cache_enc,
        )

        return predictions, sampler_state
    
    def update_sampler_state(self, sampler_state, last_subtoken, ctx_V_enc):
        
        (
        P_tT,
        cvrs_logprobs,
        covers,
        prev_covers_encs,
        prev_context_enc,
        div_enc,
        ) = extract_sampler_state(sampler_state)

        logP_tT = torch.log(P_tT)
        new_covers, new_encs, idx_temp = [], [], []

        # Step 2a. Add all subtoken within prefix. 
        for i in self.map_subtoken_bigtoken[last_subtoken]:
            new_covers.append(self.symbols2enc(
               self.enc2symbols(i, self.orig_tokenizer), self.sub_tokenizer
            ))
            encs_to_add = prev_context_enc[len(div_enc) : ] + [i]
            new_encs.append(encs_to_add)
            idx_temp.append(i)

        new_logprobs = logP_tT[idx_temp]
        if len(new_logprobs) > 0:
            cvrs_logprobs = torch.cat((cvrs_logprobs, new_logprobs))
            covers = covers + new_covers
            covers_encs = prev_covers_encs + new_encs
        else:
            covers_encs = prev_covers_encs

        # Step 2b. Remove all the unrelated tokens
        tmp_covers, tmp_cvrs_logprobs, tmp_cover_encs = [], [], []
        for k, cover_subenc in enumerate(covers):
            if len(cover_subenc) == 0:
                continue 
            if cover_subenc[0] == last_subtoken:
                tmp_covers.append(cover_subenc[1:])
                tmp_cvrs_logprobs.append(cvrs_logprobs[k])
                tmp_cover_encs.append(covers_encs[k])
        covers = tmp_covers
        cvrs_logprobs = torch.stack(tmp_cvrs_logprobs)
        covers_encs = tmp_cover_encs

        # Readjust the cover encs:
        new_loc = -1
        for i in range(1, len(ctx_V_enc)):
            if self.enc2symbols(ctx_V_enc[i], self.orig_tokenizer)[0] == self.white_space:
                new_loc = i

        new_div_enc = ctx_V_enc[:new_loc] if new_loc > 0 else div_enc
        if len(new_div_enc) != len(div_enc):
            len_diff = len(new_div_enc) - len(div_enc)
            new_covers_encs = [c[len_diff:] for c in covers_encs]
        else:
            new_covers_encs = covers_encs
        
        return (
            cvrs_logprobs,
            covers,
            new_covers_encs,
            new_div_enc,
        )
    
    def _run_kv(
        self,
        ctx_enc,
        sampler_state,
        div_enc,
    ):
        """
        Next-token prediction, deciding which KV cache to use.
        Compute log P(t_{i+1}| t^i_1).

        Args:
            ctx_enc (List[int]): context encoding.
            sampler_state (SamplerState): meta data from previous runs, contains cover(x^{n-1}) and KV caches.
            temperature (float): temperature to scale probability.
            div_enc (List[int]): encodings end before the last whitespace.

        Returns:
            log_pt_T (torch.Tensor): log P(t_{i+1}| t^i_1) for all t_{i+1}
            mask (torch.Tensor): invalid encoding masks.
            cache : KV cache of t^i.
        """
        if sampler_state != None:
            new_cache_enc = sampler_state.new_cache_enc
            if new_cache_enc == div_enc:
                sampler_state.old_cache = sampler_state.new_cache
                sampler_state.old_cache_enc = sampler_state.new_cache_enc
            else:
                assert sampler_state.old_cache_enc == div_enc
                assert sampler_state.old_cache != None
            cache = sampler_state.old_cache
            cache_enc = sampler_state.old_cache_enc
        else:
            cache = None
            cache_enc = []

        log_probs, cache = self.logprobs(
            input_ids=ctx_enc, past_key_values=cache, use_cache=True, cache_enc=cache_enc 
        )

        return log_probs, cache