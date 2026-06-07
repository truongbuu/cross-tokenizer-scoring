from dataclasses import dataclass
from typing import Optional, Tuple, List

import copy
import torch
import numpy as np
import regex as re

from src.base_instructmodel import BaseModel
from transformers import AutoTokenizer


IM_START_TOKEN_ID = 151644
SPECIAL_TOKEN_START = 151643
SPECIAL_TOKEN_COUNT = 22
SPACE_SUBTOKEN_ID = 220


def truncate_invalid_suffix(b: bytes):
    """
    From the right, find the longest valid UTF-8 prefix of `b`.
    Replace the entire invalid suffix (if any) with a single `replacement_byte`.
    """
    # Start from the end, move left until we find a valid UTF-8 decoding
    for pos in range(len(b), -1, -1):
        try:
            b[:pos].decode("utf-8")
            if pos == len(b):
                return b, None
            return b[:pos] + b"a", b[pos:]
        except UnicodeDecodeError:
            continue
    # If nothing is valid (shouldn't happen if prefix is known good)
    return b"a", b

def is_all_g(s: str) -> bool:
    return len(s) > 0 and all(ch == "Ġ" for ch in s)


def is_valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def split_by_last_space_marker(s: str):
    """
    Return (before_last_run, last_run_and_rest), where the split occurs at the
    last consecutive run of 'Ġ' characters. If no 'Ġ' is found, returns (s, "").
    """
    i = s.rfind("Ġ")
    if i == -1:
        return s, ""

    j = i
    while j > 0 and s[j - 1] == "Ġ":
        j -= 1

    return s[:j], s[j:]


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
    instruct_message_Vsub: Optional[List[int]]
    instruct_message_V: Optional[List[int]]
    P_tT2:  Optional[List[int]]
    cvrs_logprobs2:  Optional[List[int]]
    covers2:  Optional[List[int]]
    cover_encs2:  Optional[List[int]]
    prev_context_enc2:  Optional[List[int]]


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
    instruct_message_Vsub = sampler_state.instruct_message_Vsub
    instruct_message_V = sampler_state.instruct_message_V


    #assert prev_context_enc[:len(div_enc)] == div_enc

    return (
        P_tT,
        cvrs_logprobs,
        covers,
        cover_encs,
        prev_context_enc,
        div_enc,
        instruct_message_Vsub,
        instruct_message_V
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
        sub_vocab_dir: str = "Qwen2",
        vocab_file="Qwen2/vocab.json",
        merge_file="Qwen2/merges.txt"
    ):
        super().__init__(
            pretrained_model_name_or_path,
            device,
            vocab_file=vocab_file,
            merge_file=merge_file,
        )

        self.sub_tokenizer = AutoTokenizer.from_pretrained(
            sub_vocab_dir,
            use_fast=False,
            trust_remote_code=True,
        )
        self.device = device

        self.white_space = 'Ġ'

        # Build trie of subtoken sequences
        self.create_trie()

        # Create sparse prefix matrix
        self.create_prefix_mat()

        # Create map subtoken --> big token
        self.create_sub2bigtoken()
        self.whitespace_maniac()
        special_tokens = [
            self.orig_tokenizer.decode(i)
            for i in range(SPECIAL_TOKEN_START, SPECIAL_TOKEN_START + SPECIAL_TOKEN_COUNT)
        ]
        VM_special_tokens_id = self.orig_tokenizer.encode(special_tokens)
        Vsub_special_tokens_id = self.sub_tokenizer.encode(special_tokens)
        self.mapping_special_tokens = dict(
            zip(Vsub_special_tokens_id, VM_special_tokens_id)
        )
        self.VM_special_tokens_id = set(VM_special_tokens_id)

    def _token_to_subtokens(self, token_id: int) -> List[int]:
        token = self.orig_tokenizer.convert_ids_to_tokens(token_id)
        return self.sub_tokenizer.convert_tokens_to_ids(
            self.sub_tokenizer.bpe(token).split(" ")
        )

    def create_trie(self):
        self.trie = {}
        for i in range(self.orig_tokenizer.vocab_size):
            insert_sequence(self.trie, self._token_to_subtokens(i))

    def whitespace_maniac(self):
        list_whitespace = {}
        for i in range(self.orig_tokenizer.vocab_size):
            token = self.orig_tokenizer.convert_ids_to_tokens(i)
            if is_all_g(token):
                list_whitespace[i] = len(token)
        sorted_dict = dict(sorted(list_whitespace.items(), key=lambda x: x[1]))
        self.list_whitespace = sorted_dict


    def create_prefix_mat(self):
        prefix_mat = torch.zeros(
            self.sub_tokenizer.vocab_size,
            self.orig_tokenizer.vocab_size,
            dtype=self.sparse_dtype,
        )

        for i in range(self.orig_tokenizer.vocab_size):
            sub_tokens = self._token_to_subtokens(i)
            prefix_mat[sub_tokens[0], i] = 1

        indices = torch.nonzero(prefix_mat, as_tuple=False).t()
        values = prefix_mat[indices[0], indices[1]]
        self.sparse_prefix_mat = torch.sparse_coo_tensor(
            indices,
            values.to(dtype=self.sparse_dtype),
            prefix_mat.shape
            ).to(self.llm.device)

    def create_sub2bigtoken(self):
        self.map_subtoken_bigtoken = {}
        for i in range(self.orig_tokenizer.vocab_size):
            sub_tokens = self._token_to_subtokens(i)
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

    def subencode_instruct(self, prompt):
        encoded_prompt = self.sub_tokenizer.encode(prompt)
        for i in range(len(encoded_prompt)):
            if encoded_prompt[i] in self.mapping_special_tokens.keys():
                encoded_prompt[i] = self.mapping_special_tokens[encoded_prompt[i]]

        return encoded_prompt

    def enc2symbols(self, encodings, tokenizer):
        # Convert encodings to symbols format.
        # Note that symbols is different from string. Like the white space thing.
        symbols = "".join(tokenizer.convert_ids_to_tokens(encodings))
        return symbols

    def symbols2enc(self, symbols, tokenizer):
        # Convert list of symbols into list of token ids
        encs = tokenizer.convert_tokens_to_ids(tokenizer.bpe(symbols).split(" "))
        return encs

    def _cover_subtokens_for_main_token(self, token_id: int) -> List[int]:
        return self.symbols2enc(
            self.enc2symbols(token_id, self.orig_tokenizer),
            self.sub_tokenizer,
        )

    def _advance_covers(
        self,
        last_subtoken,
        div_enc,
        P_tT,
        cvrs_logprobs,
        covers,
        prev_covers_encs,
        prev_context_enc,
    ):
        logP_tT = torch.log(P_tT)
        new_covers, new_encs, idx_temp = [], [], []

        for token_id in self.map_subtoken_bigtoken[last_subtoken]:
            new_covers.append(self._cover_subtokens_for_main_token(token_id))
            new_encs.append(prev_context_enc[len(div_enc):] + [token_id])
            idx_temp.append(token_id)

        new_logprobs = logP_tT[idx_temp]
        if len(new_logprobs) > 0:
            cvrs_logprobs = torch.cat((cvrs_logprobs, new_logprobs))
            covers = covers + new_covers
            covers_encs = prev_covers_encs + new_encs
        else:
            covers_encs = prev_covers_encs

        matched_covers, matched_logprobs, matched_encs = [], [], []
        for k, cover_subenc in enumerate(covers):
            if cover_subenc and cover_subenc[0] == last_subtoken:
                matched_covers.append(cover_subenc[1:])
                matched_logprobs.append(cvrs_logprobs[k])
                matched_encs.append(covers_encs[k])

        if not matched_logprobs:
            raise ValueError(f"No cover encodings matched subtoken {last_subtoken}")

        return torch.stack(matched_logprobs), matched_covers, matched_encs

    def _cover_transition_matrix(self, p_atom_T, covers, dtype=None):
        dtype = dtype or self.sparse_dtype
        mat_sum = torch.zeros(len(p_atom_T), len(covers), dtype=dtype)
        first_subtokens = [[sub[0], k] for k, sub in enumerate(covers) if sub]
        if first_subtokens:
            temp_array = np.array(first_subtokens)
            mat_sum[temp_array[:, 0], temp_array[:, 1]] = 1.0
        return mat_sum.to(self.device)

    def _find_new_div_enc(self, ctx_V_enc, div_enc):
        if ctx_V_enc[-1] == IM_START_TOKEN_ID:
            return div_enc

        new_loc = -1
        for i in range(1, len(ctx_V_enc) - 1):
            current = self.enc2symbols(ctx_V_enc[i], self.orig_tokenizer)
            previous = self.enc2symbols(ctx_V_enc[i - 1], self.orig_tokenizer)
            following = self.enc2symbols(ctx_V_enc[i + 1], self.orig_tokenizer)
            if (
                current[0] == self.white_space
                and current[-1] != self.white_space
                and previous[-1] != self.white_space
                and following[-1] != self.white_space
            ):
                new_loc = i

        candidate = ctx_V_enc[:new_loc] if new_loc > 0 else div_enc
        return candidate if len(candidate) > len(div_enc) else div_enc

    def sub2mainV(self, raw_subencoding):
        specials = set(self.VM_special_tokens_id)

        segment = {}
        segment_num = 0
        last_special_idx = None

        for i, tid in enumerate(raw_subencoding):
            if tid in specials:
                if last_special_idx is not None:
                    # include the starting special token in the segment
                    start = last_special_idx
                    end = i
                    if start < end:  # skip empty slices
                        segment[segment_num] = raw_subencoding[start:end]
                        segment_num += 1
                last_special_idx = i

        # Also grab tail after the final special (inclusive of it)
        if last_special_idx is not None and last_special_idx < len(raw_subencoding):
            segment[segment_num] = raw_subencoding[last_special_idx:]

        #recheck step
        recheck = []
        for k in segment.keys():
            recheck = recheck + segment[k]
        assert recheck == raw_subencoding

        #token segment
        new_token_sequence = []
        for k in segment.keys():
            special_t = segment[k][0]
            if len(segment[k]) > 1:
                content_t = segment[k][1:]
                inp_sym = self.enc2symbols(content_t, self.sub_tokenizer)
                byte_decoder = {v: k for k, v in self.orig_tokenizer.byte_encoder.items()}
                reverted_bytes = bytes([byte_decoder[c] for c in inp_sym])
                if is_valid_utf8(reverted_bytes):
                    V_token = self.orig_tokenizer.encode(reverted_bytes.decode("utf-8"))
                else:
                    temp_bytes, incomplete_bytes = truncate_invalid_suffix(reverted_bytes)
                    temp_str = temp_bytes.decode("utf-8")
                    all_small_chunks = []
                    for token in re.findall(self.orig_tokenizer.pat, temp_str):
                        all_small_chunks.append(token)
                    bpe_tokens = []
                    for chunk in range(len(all_small_chunks) -1):
                        bpe_tokens += self.orig_tokenizer.encode(all_small_chunks[chunk])

                    final_bytes = all_small_chunks[-1].encode("utf-8")[:-1] + incomplete_bytes
                    final_tokens = "".join(self.orig_tokenizer.byte_encoder[b] for b in final_bytes)
                    final_bpe_tokens = [lol for lol in self.orig_tokenizer.bpe(final_tokens).split(" ")]
                    bpe_tokens += self.orig_tokenizer.convert_tokens_to_ids(final_bpe_tokens)
                    V_token = bpe_tokens
                new_token_sequence = new_token_sequence + [special_t] + V_token
            else:
                new_token_sequence = new_token_sequence + [special_t]
        return new_token_sequence

    def resolving_whitespace(self, raw_subencoding, div_enc, P_tT, cvrs_logprobs, covers, prev_covers_encs, prev_context_enc, ctx_V_enc):
        last_subtoken = raw_subencoding[-1]
        return self._advance_covers(
            last_subtoken,
            div_enc,
            P_tT,
            cvrs_logprobs,
            covers,
            prev_covers_encs,
            prev_context_enc,
        )


    def processing_whitespace_maniac(self, raw_subencoding, ctx_V_enc1, ctx_V_enc2, prev_sampler_state):
        # white space so we would avoid storing any new
        # We accept two cover encodings.

        (
        P_tT,
        cvrs_logprobs,
        covers,
        prev_covers_encs,
        prev_context_enc,
        div_enc,
        instruct_message_Vsub,
        instruct_message_V
        ) = extract_sampler_state(prev_sampler_state)

        if len(P_tT) > 1:
            return self.double_processing_whitespace_maniac(raw_subencoding, ctx_V_enc1, ctx_V_enc2, prev_sampler_state)


        baselogprob4enc2 = cvrs_logprobs[0][prev_covers_encs[0].index(ctx_V_enc2[len(div_enc):-1])]

        cvrs_logprobs1, covers1, covers_encs1 = self.resolving_whitespace(raw_subencoding, div_enc, P_tT[0], cvrs_logprobs[0], covers[0], prev_covers_encs[0], prev_context_enc[0], ctx_V_enc1)
        log_pt_T1, _ = self.logprobs(input_ids=ctx_V_enc1, past_key_values=prev_sampler_state.old_cache, use_cache=True, cache_enc=prev_sampler_state.old_cache_enc)
        #log_pt_T1 = log_pt_T1[0,-1,:]

        log_pt_T2, _ = self.logprobs(input_ids=ctx_V_enc2, past_key_values=prev_sampler_state.old_cache, use_cache=True, cache_enc=prev_sampler_state.old_cache_enc)
        #log_pt_T2 = log_pt_T2[0,-1,:]

        our_predictions = torch.zeros_like(log_pt_T1)[0,0]
        log_prob_special_tokens = log_pt_T1[0, -1, self.orig_tokenizer.vocab_size : ]
        our_predictions[self.orig_tokenizer.vocab_size :] = log_prob_special_tokens.exp()
        log_pt_T1 = log_pt_T1[0, -1, :self.orig_tokenizer.vocab_size]
        log_pt_T2 = log_pt_T2[0, -1, :self.orig_tokenizer.vocab_size]


        max_log_prob = cvrs_logprobs1.max()
        cvrs_logprobs1 -= max_log_prob
        baselogprob4enc2 -= max_log_prob
        cvrs_scaled_probs = torch.exp(cvrs_logprobs1)

        log_pT_V1 = cvrs_logprobs1[
            covers_encs1.index(ctx_V_enc1[len(div_enc):])
        ]
        ptT1_V = torch.exp(log_pT_V1 + log_pt_T1)
        ptT2_V = torch.exp(baselogprob4enc2 + log_pt_T2)

        p_atom_T1 = torch.sparse.mm(
            self.sparse_prefix_mat, ptT1_V.view(-1, 1)
        )

        p_atom_T2 = torch.sparse.mm(
            self.sparse_prefix_mat, ptT2_V.view(-1, 1)
        )

        P_unorm = (
            torch.matmul(
                self._cover_transition_matrix(p_atom_T1, covers1),
                cvrs_scaled_probs,
            )
            + p_atom_T1[:, 0] + p_atom_T2[:, 0]
        )
        predictions = (1-log_prob_special_tokens.exp().sum()) * P_unorm / P_unorm.sum()
        our_predictions[:len(predictions)] = predictions

        sampler_state = SamplerState(
            P_tT=ptT1_V[None,...],
            cvrs_logprobs=[cvrs_logprobs1],
            covers=[covers1],
            cover_encs=[covers_encs1],
            prev_context_enc=[ctx_V_enc1],
            div_enc=div_enc,
            old_cache=prev_sampler_state.old_cache,
            old_cache_enc=prev_sampler_state.old_cache_enc,
            new_cache=prev_sampler_state.new_cache,
            new_cache_enc=prev_sampler_state.new_cache_enc,
            instruct_message_Vsub= prev_sampler_state.instruct_message_Vsub,
            instruct_message_V= prev_sampler_state.instruct_message_V,
            P_tT2=ptT2_V[None,...],
            cvrs_logprobs2=[baselogprob4enc2],
            covers2=None,
            cover_encs2=None,
            prev_context_enc2=[ctx_V_enc2],
        )

        return our_predictions, sampler_state

    def check_valid_space_merge(self, raw_subencoding, ctx_V_enc1, ctx_V_enc):
        last_pos = -1
        for i in range(len(raw_subencoding) - 1, -1, -1):
            if self.orig_tokenizer.decode(raw_subencoding[i]) != " ":
                last_pos=i
                break
        valid = self.sub_tokenizer.encode(self.sub_tokenizer.decode(raw_subencoding[last_pos:])) == raw_subencoding[last_pos:]
        return valid

    def prob_next_subtoken(
        self,
        raw_subencoding: List[int],
        sampler_state: Optional[SamplerState] = None,
        debug=False,
    ):
        last_subtoken = raw_subencoding[-1]
        ctx_V_enc1 = self.sub2mainV(raw_subencoding)
        ctx_V_enc2 = self.sub2mainV(raw_subencoding+[16])[:-1]

        if ctx_V_enc1 != ctx_V_enc2:
            if last_subtoken != SPACE_SUBTOKEN_ID:
                # Choose the first ctx_enc
                ctx_V_enc = ctx_V_enc1
            elif not self.check_valid_space_merge(raw_subencoding, ctx_V_enc1, ctx_V_enc2):
                ctx_V_enc = ctx_V_enc2
            else:
                prob, sampler_state = self.processing_whitespace_maniac(raw_subencoding, ctx_V_enc1, ctx_V_enc2, sampler_state)
                return prob, sampler_state
        else:
            ctx_V_enc = ctx_V_enc1




        if sampler_state is None:
            assert last_subtoken == self.orig_tokenizer.encode('<|im_start|>')[0]
            cover_Vencs = [ctx_V_enc]
            log_prob_cover_Vencs = torch.Tensor([0.0]).to(self.device)
            cond_str_enc_V = copy.deepcopy(ctx_V_enc)
            cover_Vsub = [[]]
            instruct_message_Vsub = copy.deepcopy(raw_subencoding)
            instruct_message_V = copy.deepcopy(ctx_V_enc)
        else:
            (
            log_prob_cover_Vencs,
            cover_Vsub,
            cover_Vencs,
            cond_str_enc_V
            ) = self.update_sampler_state(sampler_state, last_subtoken, ctx_V_enc)
            #print (cover_Vsub, cover_Vencs)
            try:
                cover_Vencs.index(ctx_V_enc[len(cond_str_enc_V):])
            except ValueError:
                #Check if we get the right deal
                try:
                    (
                    log_prob_cover_Vencs,
                    cover_Vsub,
                    cover_Vencs,
                    cond_str_enc_V
                    ) = self.update_sampler_state_special(sampler_state, last_subtoken, ctx_V_enc)
                    cover_Vencs.index(ctx_V_enc[len(cond_str_enc_V):])
                except AssertionError:
                    print("ASSERT FAILED inside update_sampler_state_special")
                    return None, None
                except Exception as e:
                    print("Exception:", e)
                    return None, None

        log_pt_T, cache, cached_div_enc = self._run_kv(ctx_V_enc, sampler_state, cond_str_enc_V)
        try:
            our_predictions = torch.zeros_like(log_pt_T)[0,0]
        except Exception:
            return None, None
        log_prob_special_tokens = log_pt_T[0, -1, self.orig_tokenizer.vocab_size : ]
        #our_predictions[self.orig_tokenizer.vocab_size :] = log_prob_special_tokens.exp()
        log_pt_T = log_pt_T[0, -1, :self.orig_tokenizer.vocab_size]

        cvrs_scaled_probs, log_prob_cover_Vencs = log2prob_rescale(
            log_prob_cover_Vencs
        )

        # step 2:
        try:
            if sampler_state is not None:
                log_pT_V = log_prob_cover_Vencs[
                    cover_Vencs.index(ctx_V_enc[len(cond_str_enc_V):])
                ]
            else:
                log_pT_V = 0.0
        except ValueError:
            return None, None
        ptT_V = torch.exp(log_pT_V + log_pt_T).to(dtype=self.sparse_dtype)
        p_atom_T = torch.sparse.mm(
            self.sparse_prefix_mat, ptT_V.view(-1, 1)
        )

        p_atom2_T = torch.exp(log_pT_V + log_prob_special_tokens)

        P_unorm = (
            torch.matmul(
                self._cover_transition_matrix(p_atom_T, cover_Vsub),
                cvrs_scaled_probs,
            )
            + p_atom_T[:, 0]
        )


        #P_unorm= torch.cat((P_unorm, p_atom2_T), dim=0)
        our_predictions[self.orig_tokenizer.vocab_size :] = p_atom2_T
        our_predictions[:len(P_unorm)] = P_unorm
        our_predictions = our_predictions/our_predictions.sum()
        #predictions = P_unorm/P_unorm.sum()
        #(1-log_prob_special_tokens.exp().sum()) * P_unorm / P_unorm.sum()

        # Update the kv cache.
        if sampler_state is not None:
            instruct_message_Vsub = sampler_state.instruct_message_Vsub
            instruct_message_V = sampler_state.instruct_message_V

        sampler_state = SamplerState(
            P_tT=ptT_V[None,...],
            cvrs_logprobs=[log_prob_cover_Vencs],
            covers=[cover_Vsub],
            cover_encs=[cover_Vencs],
            prev_context_enc=[ctx_V_enc],
            div_enc=cond_str_enc_V,
            old_cache=cache,
            old_cache_enc=cached_div_enc,
            new_cache=cache,
            new_cache_enc=cached_div_enc,
            instruct_message_Vsub= instruct_message_Vsub,
            instruct_message_V=instruct_message_V,
            P_tT2=  None,
            cvrs_logprobs2=  None,
            covers2=  None,
            cover_encs2=  None,
            prev_context_enc2=  None
        )

        #our_predictions[:len(predictions)] = predictions
        return our_predictions, sampler_state

    def update_sampler_state(self, sampler_state, last_subtoken, ctx_V_enc):
        (
        P_tT,
        cvrs_logprobs,
        covers,
        prev_covers_encs,
        prev_context_enc,
        div_enc,
        instruct_message_Vsub,
        instruct_message_V
        ) = extract_sampler_state(sampler_state)

        assert P_tT.shape[0] != 2
        P_tT = P_tT[0]
        cvrs_logprobs = cvrs_logprobs[0]
        covers = covers[0]
        prev_covers_encs = prev_covers_encs[0]
        prev_context_enc = prev_context_enc[0]

        cvrs_logprobs, covers, covers_encs = self._advance_covers(
            last_subtoken,
            div_enc,
            P_tT,
            cvrs_logprobs,
            covers,
            prev_covers_encs,
            prev_context_enc,
        )

        new_div_enc = self._find_new_div_enc(ctx_V_enc, div_enc)
        if len(new_div_enc) != len(div_enc):
            len_diff = len(new_div_enc) - len(div_enc)
            new_covers_encs = [c[len_diff:] for c in covers_encs]
        else:
            new_covers_encs = covers_encs

        if (
            len(new_covers_encs) > 1
            and new_covers_encs[0] == []
            and new_covers_encs[1] == []
        ):
            raise ValueError(
                "Ambiguous empty cover encodings after sampler-state update"
            )

        return (
            cvrs_logprobs,
            covers,
            new_covers_encs,
            new_div_enc,
        )

    def update_sampler_state_special(self, sampler_state, last_subtoken, ctx_V_enc):
        P_tT = sampler_state.P_tT2
        prev_context_enc = sampler_state.prev_context_enc2
        div_enc = sampler_state.div_enc

        assert P_tT.shape[0] != 2

        P_tT = P_tT[0]
        prev_context_enc = prev_context_enc[0]

        logP_tT = torch.log(P_tT)
        new_covers, new_encs, idx_temp = [], [], []

        # Step 2a. Add all subtoken within prefix.
        for token_id in self.map_subtoken_bigtoken[last_subtoken]:
            new_covers.append(self._cover_subtokens_for_main_token(token_id))
            new_encs.append(prev_context_enc[len(div_enc):] + [token_id])
            idx_temp.append(token_id)

        new_logprobs = logP_tT[idx_temp]
        assert len(new_logprobs) > 0

        return (
            new_logprobs,
            new_covers,
            new_encs,
            sampler_state.div_enc,
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
        if sampler_state is not None:
            old_cache_enc = sampler_state.new_cache_enc
            old_cache = sampler_state.new_cache
            if div_enc != old_cache_enc:
                _, cache = self.logprobs(
                    input_ids=div_enc, past_key_values=old_cache, use_cache=True, cache_enc=old_cache_enc
                )
            else:
                cache = old_cache
            log_probs, _ = self.logprobs(
                input_ids=ctx_enc, past_key_values=cache, use_cache=True, cache_enc=div_enc
            )
        else:
            # prefill to div_enc
            old_cache = None
            old_cache_enc = []

            assert ctx_enc == div_enc

            log_probs, cache = self.logprobs(
                input_ids=ctx_enc, past_key_values=old_cache, use_cache=True, cache_enc=old_cache_enc
            )

        return log_probs, cache, div_enc
