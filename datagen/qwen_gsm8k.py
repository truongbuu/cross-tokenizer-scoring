#!/usr/bin/env python
import sys
import json
import random
import numpy as np
from datasets import Dataset, load_dataset
import torch
import argparse
import tqdm as tqdm

# ========== Args ==========
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=0, help="Subset vocab size")
parser.add_argument("--model_dir", type=str, default="./Qwen2.5-0.5B-Instruct/",
                    help="Path to the model directory")
parser.add_argument("--split", type=int, default=0, help="Split")
args = parser.parse_args()

# ========== Your model/tokenizer import ==========
sys.path.append("./src")
from src.Qwen2_subvocab_model_instruct import SubVocabPredLLM

# ========== Config ==========
MODEL_DIR = args.model_dir
SUB_VOCAB_DIR = f"./qwen_vocabs/subset_vocabs/subsetQwen2.5Instruct_{args.n}"
VOCAB_FILE = MODEL_DIR + "vocab.json"
MERGE_FILE = MODEL_DIR + "merges.txt"
DATA_SPLIT = f"splits_jsonl_gsm8k/gsm8k_main_train_part_{args.split}.jsonl"
#f"./splits_jsonl_gsm8k/alpaca_part_{args.split}.jsonl"

print(f"Using MODEL_DIR = {MODEL_DIR}")
print(f"Using VOCAB_FILE = {VOCAB_FILE}")
print(f"Using MERGE_FILE = {MERGE_FILE}")


ASSIST_START_TOKEN_ID = 151644   # <im_start>
N_EXAMPLES = 16                  # take first N examples
TOPK = 20
TRUNCATE_TO = None               # e.g., 2048 to cap sequence length; or None
JSONL_OUT = f"./out_jsonl{args.n}_gsm8k/topk_gsm8k_part_{args.split}.jsonl"

SEED = 0
rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

# ========== Helpers ==========
def format_example(example):
    #import pdb; pdb.set_trace()
    instruction = (example.get("question", "") or "").strip()
    input_text  = "" #(example.get("input", "") or "").strip()
    output      = (example.get("answer", "") or "").strip()
    user_message = f"{instruction}\n\n{input_text}" if input_text else instruction
    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output}
        ]
    }

def get_vocab_size(tok):
    if hasattr(tok, "vocab_size") and tok.vocab_size is not None:
        return int(tok.vocab_size)
    try:
        return len(tok.get_vocab())
    except Exception:
        raise ValueError("Cannot determine vocab size from tokenizer")

def last_index_of(xs, value):
    """Return the last index of `value` in list `xs`, or -1 if not present."""
    for i in range(len(xs) - 1, -1, -1):
        if xs[i] == value:
            return i
    return -1

def truncate_to_decimals(x: torch.Tensor, decimals=3):
    """Truncate each element in x to given number of decimal places."""
    factor = 10.0 ** decimals
    return torch.trunc(x * factor) / factor
#==================================================================================

def make_label(messages, model, k=TOPK, truncate_to=TRUNCATE_TO):
    # 1) render chat to a single string with your chat template
    #import pdb; pdb.set_trace()
    text = model.sub_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    # 2) sub-encode to token ids (list[int])
    ids = model.subencode_instruct(text)
    ids = ids[:-1]

    # optional truncation (keep at least 2 tokens for (x,y) pairs)
    if truncate_to is not None and len(ids) > truncate_to:
        ids = ids[:truncate_to]
    if len(ids) < 2:
        return None

    # ---- Supervision starts AFTER the last <im_start> (assistant) ----
    start_idx = last_index_of(ids, ASSIST_START_TOKEN_ID)
    if start_idx == -1:
        # no assistant start marker => skip
        return None
    if start_idx >= len(ids) - 1:
        # nothing to predict after <im_start>
        return None
    generation_prefix = model.sub_tokenizer.encode('assistant\n')
    begin_conv = start_idx
    start_idx = start_idx + len(generation_prefix)

    # We feed input_ids = ids[:-1] (as usual) and predict labels = ids[1:],
    # but mask everything BEFORE start_idx with -100 so that the first *kept* label
    # is ids[start_idx+1] (i.e., the token right after <im_start>).
    input_ids = ids[:-1]
    labels    = ids[1:]
    loss_mask = [0] * len(input_ids)  # 0 before start, 1 from start_idx onward


    # Mask labels before start_idx
    for i in range(start_idx):
        if i < len(labels):
            labels[i] = -100
    for i in range(start_idx, len(loss_mask)):
        loss_mask[i] = 1

    # 3) top-k per position; for masked positions we fill dummies (zeros)
    topk_ids = []
    topk_probs = []
    #import pdb; pdb.set_trace()
    sampler_state = None
    sub_encs = input_ids[:begin_conv+1]
    for i in range(len(generation_prefix)):
        preds, sampler_state = model.prob_next_subtoken(sub_encs, sampler_state=sampler_state)
        if preds == None:
            return None
        sub_encs = input_ids[:begin_conv+1+i+1]
    preds, sampler_state = model.prob_next_subtoken(sub_encs, sampler_state=sampler_state)
    #import pdb; pdb.set_trace()
    for i, y in enumerate(labels):
        if y == -100:
            topk_ids.append([-1] * k)
            topk_probs.append([-1.0] * k)
            continue
        if preds == None:
            return None
        topk_p, topk_indices = torch.topk(preds, TOPK)

        topk_ids.append(topk_indices.tolist())
        topk_probs.append(topk_p.tolist())

        if y == 151645:
            break
        sub_encs = sub_encs + [y]
        #print (model.orig_tokenizer.decode(sub_encs))
        preds, sampler_state = model.prob_next_subtoken(sub_encs, sampler_state=sampler_state)
    ex = {
        "input_ids": input_ids,    # list[int], length T-1
        "labels": labels,          # list[int], length T-1; -100 before assistant span
        "loss_mask": loss_mask,    # list[int] in {0,1}, aligns with labels
        "topk_ids": topk_ids,      # list[list[int]] (T-1, K); zeros for masked positions
        "topk_probs": topk_probs,  # list[list[float]] (T-1, K); zeros for masked positions
        "assist_start_label_idx": start_idx,  # index in labels where supervision begins
    }
    # quick consistency checks
    L = len(input_ids)
    assert len(labels) == L and len(loss_mask) == L
    assert len(topk_ids) == L and len(topk_probs) == L
    return ex

def main():
    # ---- init your sub-vocab model/tokenizer ----
    SubLLM = SubVocabPredLLM(MODEL_DIR, device="cuda", \
                             sub_vocab_dir=SUB_VOCAB_DIR,\
                             vocab_file=VOCAB_FILE,
                             merge_file=MERGE_FILE)

    vocab_size = get_vocab_size(SubLLM.sub_tokenizer)

    # ---- load and format base dataset ----
    small = load_dataset("json", data_files=DATA_SPLIT)["train"]
    #small = small.map(format_example)
    # ---- build distillation rows ----
    rows = []
    for i in tqdm.tqdm(range(len(small))):
        #import pdb; pdb.set_trace()
        messages = small[i]["messages"]
        ex = make_label(
            messages=messages,
            model=SubLLM,
            k=TOPK,
            truncate_to=TRUNCATE_TO
            )
        #import pdb; pdb.set_trace()
        if ex is not None:
            rows.append(ex)

    if not rows:
        print("No usable examples found.")
        return

    distil_ds = Dataset.from_list(rows)

    # ---- save JSONL ----
    distil_ds.to_json(JSONL_OUT, lines=True)
    print(f"Saved {len(distil_ds)} examples to {JSONL_OUT}")

    # ---- how to load it back (sanity print) ----
    reloaded = load_dataset("json", data_files={"train": JSONL_OUT})
    print(reloaded)
    ex0 = reloaded["train"][0]
    print(ex0.keys())
    print("len(input_ids) =", len(ex0["input_ids"]))
    print("len(labels)    =", len(ex0["labels"]))
    print("assist_start_label_idx =", ex0["assist_start_label_idx"])
    print("loss_mask sum  =", sum(ex0["loss_mask"]))
    print("topk row[assist_start_label_idx] len =", len(ex0["topk_ids"][ex0["assist_start_label_idx"]]))


if __name__ == "__main__":
    main()
