#!/usr/bin/env python
"""Generate Gemma-tokenized GSM8K distillation rows from Qwen subtoken scores."""

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import HybridCache

sys.path.append("./src")
from src.Qwen2_subvocab_model_instruct import SubVocabPredLLM


QWEN_SPECIAL_START = 151643
QWEN_EOS_TOKEN_ID = 151642
QWEN_ASSISTANT_END_TOKEN_ID = 151644
GEMMA_EOS_TOKEN_ID = 107
GEMMA_SPACE_TOKEN_ID = 235248
BYTE_SPACE_ID = 220
GEMMA_ASSISTANT_PATTERN = [106, 2516, 108]
PROB_THRESHOLD = 0.1
EPS = 1e-5

MASKING_BYTE_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 201, 220,
    222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
    232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 255,
]

UNMASKING_BYTE_IDS = [
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 202, 203, 204, 205, 206, 207, 208,
    209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
    219, 221,
]


@dataclass(frozen=True)
class Config:
    qwen_dir: str
    gemma_dir: str
    split: int

    @property
    def sub_vocab_dir(self):
        return "./qwen_vocabs/subset_vocabs/subsetQwen2.5Instruct_0"

    @property
    def vocab_file(self):
        return str(Path(self.qwen_dir) / "vocab.json")

    @property
    def merge_file(self):
        return str(Path(self.qwen_dir) / "merges.txt")

    @property
    def data_split(self):
        return f"./splits_jsonl_gsm8k_80/gsm8k_main_train_part_{self.split}.jsonl"

    @property
    def jsonl_out(self):
        return f"./out_jsonl_gsm8k_gemma/gsm8k_main_train_part_{self.split}.jsonl"

    @property
    def error_out(self):
        return f"./out_jsonl_gsm8k_gemma/error_gsm8k_main_train_part_{self.split}.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_dir", default="./Qwen2.5-0.5B-Instruct/", help="Path to the Qwen model directory.")
    parser.add_argument("--gemma_dir", default="google/gemma-2-2b-it", help="Path or Hugging Face ID for Gemma.")
    parser.add_argument("--split", type=int, default=0, help="GSM8K split index.")
    return parser.parse_args()


def longest_sublist_ending_with_target(xs, target=QWEN_ASSISTANT_END_TOKEN_ID):
    try:
        end_idx = len(xs) - 1 - xs[::-1].index(target)
        return xs[: end_idx + 1]
    except ValueError:
        return []


def find_subsequence_indices(sequence, pattern=None):
    pattern = GEMMA_ASSISTANT_PATTERN if pattern is None else pattern
    matches = []
    width = len(pattern)
    for idx in range(len(sequence) - width + 1):
        if sequence[idx : idx + width] == pattern:
            matches.append(idx)
    return matches


def top_tokens_above_threshold(probs, k):
    values, token_ids = torch.topk(probs, k)
    keep = values >= PROB_THRESHOLD
    return token_ids[keep], values[keep]


def build_qwen_model(config):
    return SubVocabPredLLM(
        config.qwen_dir,
        device="cuda",
        sub_vocab_dir=config.sub_vocab_dir,
        vocab_file=config.vocab_file,
        merge_file=config.merge_file,
    )


def build_gemma_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.gemma_dir)
    model = AutoModelForCausalLM.from_pretrained(
        config.gemma_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).cuda()
    return tokenizer, model


def gen_candidates(model, target_tok, input_ids, context, orig_sampler_state, n=20):
    out, sampler_state = model.prob_next_subtoken(context, sampler_state=orig_sampler_state)
    filtered_tokens, _ = top_tokens_above_threshold(out, k=3)
    candidates = []

    for token in filtered_tokens:
        token_id = token.item()
        if token_id > QWEN_EOS_TOKEN_ID:
            break

        saved_state = copy.deepcopy(sampler_state)
        gen_bytes = [token_id]
        this_context = context + [token_id]

        for _ in range(n):
            out, saved_state = model.prob_next_subtoken(this_context, sampler_state=saved_state)
            next_id = int(out.argmax().item())
            if next_id > QWEN_EOS_TOKEN_ID:
                break
            gen_bytes.append(next_id)
            this_context.append(next_id)
            if next_id == BYTE_SPACE_ID:
                break

        next_token = target_tok.encode(model.orig_tokenizer.decode(gen_bytes))[1]
        decoded = target_tok.decode(input_ids[0][1:].tolist() + [next_token])
        if input_ids[0].tolist() + [next_token] == target_tok.encode(decoded):
            candidates.append(next_token)

    """
    text = model.orig_tokenizer.decode(context)
    qwen_encs = model.orig_tokenizer.encode(text)
    logprobs, _ = model.logprobs(input_ids=qwen_encs)
    probs = logprobs.exp()[:, -1, :]
    qwen_tokens = probs.argsort()[:, -2:]

    fallback_tokens = [
        target_tok.encode(model.orig_tokenizer.decode(qwen_tokens[0, 0]))[1],
        target_tok.encode(model.orig_tokenizer.decode(qwen_tokens[0, 1]))[1],
    ]
    """
    return list(set(candidates))


def compute_special_prob(model, context, sampler_state, cond_s_logprob, cond_enc_logprob):
    out, _ = model.prob_next_subtoken(context, sampler_state=sampler_state)
    p_enc = out[QWEN_SPECIAL_START:].sum()
    log_string_x = np.log(p_enc.item() + EPS)
    log_full = np.log(p_enc.item() + EPS) + cond_s_logprob - cond_enc_logprob
    return min(0.0, log_full), min(0.0, log_string_x)


def encode_candidate_bytes(candidate, model, target_tok, context, sampler_state):
    token_text = target_tok.decode(candidate)
    byte_ids = model.sub_tokenizer.encode(token_text)
    probabilities = []

    for byte_id in byte_ids:
        out, sampler_state = model.prob_next_subtoken(context, sampler_state=sampler_state)
        context = context + [byte_id]
        probabilities.append(out[byte_id].item())

    return token_text, byte_ids, np.asarray(probabilities), context, sampler_state, out


def subtract_overlapping_merges(p_enc, probability, candidate, model, target_tok, context, sampler_state, out):
    filtered_chars, filtered_probs = top_tokens_above_threshold(out, k=4)
    candidate_len = len(target_tok.decode(candidate))

    for char, prob in zip(filtered_chars, filtered_probs):
        temp_sampler_state = copy.deepcopy(sampler_state)
        temp_context = context + [char.item()]
        next_char_prob = [prob.item()]
        gen_bytes = [char.item()]
        saw_special_id = False

        for _ in range(10):
            if temp_context[-1] > QWEN_EOS_TOKEN_ID:
                saw_special_id = True
                break

            out, temp_sampler_state = model.prob_next_subtoken(temp_context, sampler_state=temp_sampler_state)
            next_id = int(out.argmax().item())
            temp_context.append(next_id)
            gen_bytes.append(next_id)
            next_char_prob.append(out.max().item())

            if next_id == BYTE_SPACE_ID:
                break
            if next_id > QWEN_EOS_TOKEN_ID:
                saw_special_id = True
                break

        merge_char_prob = np.concatenate((probability, np.asarray(next_char_prob)))
        decoded = target_tok.decode(candidate) + model.orig_tokenizer.decode(gen_bytes[:-1] if saw_special_id else gen_bytes)

        if saw_special_id:
            gemma_enc = target_tok.encode(decoded)[1:] + [GEMMA_EOS_TOKEN_ID]
            bytes_beam = model.sub_tokenizer.encode(decoded) + [gen_bytes[-1]]
        else:
            gemma_enc = target_tok.encode(decoded)[1:]
            bytes_beam = model.sub_tokenizer.encode(decoded)

        if gemma_enc[0] == candidate:
            continue

        for prefix_len in range(1, len(gemma_enc) + 1):
            current_len = len(target_tok.decode(gemma_enc[:prefix_len]))
            if current_len < len(bytes_beam):
                if current_len > candidate_len and bytes_beam[current_len] == BYTE_SPACE_ID:
                    p_enc = p_enc - merge_char_prob[: current_len + 1].prod()
                    break
                if current_len > candidate_len and bytes_beam[current_len] > QWEN_EOS_TOKEN_ID:
                    p_enc = p_enc - merge_char_prob[: current_len + 1].prod()
                    break
            if current_len > candidate_len:
                p_enc = p_enc - merge_char_prob[:current_len].prod()
                break

    return p_enc


def compute_prob(candidate, model, target_tok, input_ids, context, orig_sampler_state, cond_s_logprob, cond_enc_logprob):
    print("candidate: ", candidate)
    sampler_state = orig_sampler_state

    if candidate <= GEMMA_EOS_TOKEN_ID:
        return compute_special_prob(model, context, sampler_state, cond_s_logprob, cond_enc_logprob)

    token_text, byte_ids, probability, context, sampler_state, out = encode_candidate_bytes(
        candidate, model, target_tok, context, sampler_state
    )
    p_enc = probability.prod()
    log_string_x = min(0.0, np.log(p_enc + EPS))

    if token_text.isdigit() and input_ids[0][-1] == GEMMA_SPACE_TOKEN_ID:
        out[MASKING_BYTE_IDS] = 0.0
        out = out / out.sum()
        return min(0.0, np.log(out[byte_ids[0]].item() + EPS)), log_string_x

    out, sampler_state = model.prob_next_subtoken(context, sampler_state=sampler_state)

    if candidate == GEMMA_SPACE_TOKEN_ID:
        p_enc = p_enc * out[UNMASKING_BYTE_IDS].sum()
        log_full = np.log(p_enc.item() + EPS) + cond_s_logprob - cond_enc_logprob
        return min(0.0, log_full), min(0.0, log_string_x)

    p_enc = subtract_overlapping_merges(p_enc, probability, candidate, model, target_tok, context, sampler_state, out)
    log_full = np.log(p_enc + EPS) + cond_s_logprob - cond_enc_logprob
    return min(0.0, log_full), log_string_x


def prepare_qwen_context(messages, model):
    qwen_prompt = model.sub_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    sub_encs = longest_sublist_ending_with_target(model.subencode_instruct(qwen_prompt))
    sampler_state = None

    for token_id in model.sub_tokenizer.encode("assistant\n"):
        _, sampler_state = model.prob_next_subtoken(sub_encs, sampler_state=sampler_state)
        sub_encs = sub_encs + [token_id]

    sampler_state.P_tT *= 0.0
    sampler_state.P_tT[0, 198] = 1.0
    return sub_encs, sampler_state


def init_example(gemma_input, assistant_start_idx):
    labels = gemma_input[1:-1]
    loss_mask = [1] * len(labels)
    topk_ids = []
    topk_probs = []

    for mask_id in range(assistant_start_idx + 2):
        labels[mask_id] = -100
        loss_mask[mask_id] = 0
        topk_ids.append([-1])
        topk_probs.append([-1.0])

    return {
        "input_ids": gemma_input[:-2],
        "labels": labels,
        "loss_mask": loss_mask,
        "topk_ids": topk_ids,
        "topk_probs": topk_probs,
        "assist_start_label_idx": assistant_start_idx + 2,
    }


def target_model_next_probs(model, input_ids, kv_cache, attention_mask, cache_position):
    with torch.inference_mode():
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).cuda()
            cache_position = torch.arange(attention_mask.shape[1], device="cuda")
            model_input_ids = input_ids
        else:
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device="cuda")], dim=-1)
            cache_position = cache_position[-1:] + 1
            model_input_ids = input_ids[:, -1:]

        outputs = model(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
            past_key_values=kv_cache,
        )
        probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)

    return probs, outputs.past_key_values, attention_mask, cache_position


def build_candidate_probs(candidate_tokens, gemma_token, qwen_model, gemma_tokenizer, input_ids, sub_encs, sampler_state, cond_s_logprob, cond_enc_logprob):
    log_probs = {}
    for candidate in candidate_tokens:
        saved_sampler_state = copy.deepcopy(sampler_state)
        log_probs[candidate] = compute_prob(
            candidate,
            qwen_model,
            gemma_tokenizer,
            input_ids,
            sub_encs,
            saved_sampler_state,
            cond_s_logprob,
            cond_enc_logprob,
        )

    if gemma_tokenizer.decode(gemma_token).isdigit():
        return log_probs, 0.0, 0.0

    cond_enc_logprob += log_probs[gemma_token][0]
    cond_s_logprob += log_probs[gemma_token][1]
    print(cond_enc_logprob, cond_s_logprob)
    return log_probs, cond_s_logprob, cond_enc_logprob


def append_candidate_row(example, log_probs, target_probs):
    token_ids = []
    token_probs = []

    for token_id, (log_prob, _) in log_probs.items():
        approx_prob = np.exp(log_prob)
        if approx_prob > 1.0:
            return False

        _ = target_probs[0, token_id].item()
        token_ids.append(token_id)
        token_probs.append(float(approx_prob))

    example["topk_ids"].append(token_ids)
    example["topk_probs"].append(token_probs)
    return True


def make_label(messages, qwen_model, target_model, gemma_tokenizer):
    gemma_input = gemma_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    assistant_matches = find_subsequence_indices(gemma_input)
    if not assistant_matches:
        return None

    assistant_start_idx = assistant_matches[0]
    gemma_answer = gemma_input[assistant_start_idx + 3 :]
    gemma_context = gemma_input[: assistant_start_idx + 3]
    example = init_example(gemma_input, assistant_start_idx)
    sub_encs, sampler_state = prepare_qwen_context(messages, qwen_model)

    cond_s_logprob = 0.0
    cond_enc_logprob = 0.0
    kv_cache = HybridCache(
        config=target_model.config,
        max_batch_size=1,
        max_cache_len=2048,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention_mask = None
    cache_position = None

    for gemma_token in gemma_answer:
        token_text = gemma_tokenizer.decode(gemma_token)
        if gemma_tokenizer.encode(token_text)[1] != gemma_token:
            return None

        qwen_bytes = qwen_model.sub_tokenizer.encode(token_text)
        input_ids = torch.tensor([gemma_context], dtype=torch.long).cuda()
        target_probs, kv_cache, attention_mask, cache_position = target_model_next_probs(
            target_model, input_ids, kv_cache, attention_mask, cache_position
        )
        gemma_context += [gemma_token]

        candidate_tokens = gen_candidates(qwen_model, gemma_tokenizer, input_ids, sub_encs, copy.deepcopy(sampler_state))
        candidate_tokens = list(set(candidate_tokens + [gemma_token] + [target_probs.argmax().item()]))
        log_probs, cond_s_logprob, cond_enc_logprob = build_candidate_probs(
            candidate_tokens,
            gemma_token,
            qwen_model,
            gemma_tokenizer,
            input_ids,
            sub_encs,
            sampler_state,
            cond_s_logprob,
            cond_enc_logprob,
        )

        if not append_candidate_row(example, log_probs, target_probs):
            return None
        if gemma_token == GEMMA_EOS_TOKEN_ID:
            break

        for byte_id in qwen_bytes:
            _, sampler_state = qwen_model.prob_next_subtoken(sub_encs, sampler_state=sampler_state)
            sub_encs = sub_encs + [byte_id]

    return example


def save_jsonl(rows, path, label):
    if not rows:
        return
    dataset = Dataset.from_list(rows)
    dataset.to_json(path, lines=True)
    print(f"Saved {len(dataset)} {label} to {path}")


def print_sanity_check(jsonl_path):
    reloaded = load_dataset("json", data_files={"train": jsonl_path})
    print(reloaded)
    ex0 = reloaded["train"][0]
    print(ex0.keys())
    print("len(input_ids) =", len(ex0["input_ids"]))
    print("len(labels)    =", len(ex0["labels"]))
    print("assist_start_label_idx =", ex0["assist_start_label_idx"])
    print("loss_mask sum  =", sum(ex0["loss_mask"]))
    print("topk row[assist_start_label_idx] len =", len(ex0["topk_ids"][ex0["assist_start_label_idx"]]))


def main():
    args = parse_args()
    config = Config(qwen_dir=args.qwen_dir, gemma_dir=args.gemma_dir, split=args.split)

    qwen_model = build_qwen_model(config)
    gemma_tokenizer, gemma_model = build_gemma_model(config)
    dataset = load_dataset("json", data_files=config.data_split)["train"]

    rows = []
    error_rows = []
    for idx in range(len(dataset)):
        print(f"Split {config.split}, Progress {idx}/{len(dataset)}")
        messages = dataset[idx]["messages"]
        try:
            example = make_label(messages, qwen_model, gemma_model, gemma_tokenizer)
        except Exception as err:
            print(f"Skipping example {idx}: {err}")
            example = None

        if example is None:
            error_rows.extend(messages)
        else:
            rows.append(example)

    save_jsonl(error_rows, config.error_out, "error examples")
    if not rows:
        print("No usable examples found.")
        return

    save_jsonl(rows, config.jsonl_out, "examples")
    print_sanity_check(config.jsonl_out)


if __name__ == "__main__":
    main()
