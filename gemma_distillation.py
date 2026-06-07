#!/usr/bin/env python
"""Train a Gemma student with sparse top-k distillation targets."""

import argparse
import os
import pdb
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "google/gemma-2-2b-it"
DEFAULT_DATASET_PATH = str(REPO_DIR / "out_jsonl_gsm8k_gemma" / "gsm8k_gemma_filtered.jsonl")
DEFAULT_OUTPUT_DIR = str(REPO_DIR / "outputs" / "gemma2-distil-topk_gsm8k")
DEFAULT_SAVE_DIR = str(REPO_DIR / "outputs")
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Config:
    model_id: str
    dataset_path: str
    output_dir: str
    save_root: str
    kl_weight: float
    ce_weight: float
    nepochs: int
    lr: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    bf16: bool
    alm: bool
    debug: bool
    use_lora: bool
    validation_ratio: float
    seed: int

    @property
    def final_save_dir(self) -> str:
        return os.path.join(
            self.save_root,
            (
                "gemma2-distil-topk-final_"
                f"kl{self.kl_weight}_ce{self.ce_weight}_"
                f"alm{self.alm}_epochs{self.nepochs}"
            ),
        )


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Path or Hugging Face model ID for the student model.")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH, help="JSONL distillation dataset path.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Trainer checkpoint/output directory.")
    parser.add_argument("--save_root", default=DEFAULT_SAVE_DIR, help="Directory where the final model directory is created.")
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--nepochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 training.")
    parser.add_argument("--alm", action="store_true", help="Use the ALM sparse KL objective.")
    parser.add_argument("--debug", action="store_true", help="Enable CUDA_LAUNCH_BLOCKING and post-mortem pdb.")
    parser.add_argument("--use_lora", action="store_true", help="Enable the LoRA config defined in this script.")
    parser.add_argument("--validation_ratio", type=float, default=0.05, help="Fraction of rows to hold out for eval loss. Set 0 to disable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split and Trainer.")
    args = parser.parse_args()

    if not 0.0 <= args.validation_ratio < 1.0:
        parser.error("--validation_ratio must be in [0, 1).")

    return Config(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        save_root=args.save_root,
        kl_weight=args.kl_weight,
        ce_weight=args.ce_weight,
        nepochs=args.nepochs,
        lr=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=not args.no_bf16,
        alm=args.alm,
        debug=args.debug,
        use_lora=args.use_lora,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )


def install_debug_hook() -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    def post_mortem_hook(exc_type, value, tb):
        traceback.print_exception(exc_type, value, tb)
        pdb.post_mortem(tb)

    sys.excepthook = post_mortem_hook


class TopKDistilDataset(Dataset):
    """Exposes pre-tokenized distillation examples from a Hugging Face split."""

    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        ex = self.data[idx]
        return {
            "input_ids": ex["input_ids"],
            "labels": ex["labels"],
            "loss_mask": ex["loss_mask"],
            "topk_ids": ex["topk_ids"],
            "topk_probs": ex["topk_probs"],
        }


@dataclass
class DistilCollator:
    pad_token_id: int
    max_length: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        input_ids, labels, attention_mask, loss_mask = [], [], [], []
        topk_ids, topk_probs = [], []

        for ex in batch:
            ids = ex["input_ids"][:max_len]
            labs = ex["labels"][:max_len]
            lmask = ex["loss_mask"][:max_len]
            tk_ids = ex["topk_ids"][:max_len]
            tk_probs = ex["topk_probs"][:max_len]

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                labs = labs + [IGNORE_INDEX] * pad_len
                lmask = lmask + [0] * pad_len
                tk_ids = tk_ids + [[-1]] * pad_len
                tk_probs = tk_probs + [[0.0]] * pad_len

            input_ids.append(ids)
            labels.append(labs)
            loss_mask.append(lmask)
            attention_mask.append([1 if token != self.pad_token_id else 0 for token in ids])
            topk_ids.append(tk_ids)
            topk_probs.append(tk_probs)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            "topk_ids": topk_ids,
            "topk_probs": topk_probs,
        }


def sparse_kl_loss_from_topk(logits, topk_ids, topk_probs, valid_mask):
    """Sparse KL over variable-width top-k teacher probabilities."""
    logp = F.log_softmax(logits, dim=-1)
    loss = logits.new_tensor(0.0)
    denom = valid_mask.sum().clamp_min(1)

    for batch_idx, batch_topk_ids in enumerate(topk_ids):
        for pos_idx, ids in enumerate(batch_topk_ids):
            if not valid_mask[batch_idx, pos_idx] or ids == [-1]:
                continue

            id_tensor = torch.tensor(ids, device=logits.device, dtype=torch.long)
            q = torch.tensor(topk_probs[batch_idx][pos_idx], device=logits.device, dtype=logits.dtype)
            q_sum = q.sum()
            if q_sum > 1.0:
                q = q / q_sum.clamp_min(1e-12)
                q_rest = logits.new_tensor(0.0)
            else:
                q_rest = 1.0 - q_sum

            selected_logp = logp[batch_idx, pos_idx, id_tensor]
            selected_p = selected_logp.exp().sum().clamp(max=1.0 - 1e-5)
            rest_logp = torch.log1p(-selected_p)
            loss = loss + (q * selected_logp).sum() + q_rest * rest_logp

    return -loss / denom


def sparse_kl_loss_from_topk_alm(logits, topk_ids, topk_probs, valid_mask, labels):
    """ALM variant that keeps only the ground-truth label probability when present."""
    logp = F.log_softmax(logits, dim=-1)
    loss = logits.new_tensor(0.0)
    denom = valid_mask.sum().clamp_min(1)

    for batch_idx, batch_topk_ids in enumerate(topk_ids):
        for pos_idx, ids in enumerate(batch_topk_ids):
            if not valid_mask[batch_idx, pos_idx] or ids == [-1]:
                continue

            label_id = int(labels[batch_idx, pos_idx].item())
            if label_id == IGNORE_INDEX:
                continue

            q_label = logits.new_tensor(0.0)
            for token_id, prob in zip(ids, topk_probs[batch_idx][pos_idx]):
                if token_id == label_id:
                    q_label = logits.new_tensor(prob)
                    break

            label_logp = logp[batch_idx, pos_idx, label_id]
            label_p = label_logp.exp().clamp(max=1.0 - 1e-5)
            rest_logp = torch.log1p(-label_p)
            loss = loss + q_label * label_logp + (1.0 - q_label) * rest_logp

    return -loss / denom


class TopKDistilTrainer(Trainer):
    def __init__(
        self,
        *args,
        kl_weight=1.0,
        ce_weight=0.0,
        ignore_index=IGNORE_INDEX,
        alm=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.alm = alm

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        loss_mask = inputs["loss_mask"].bool()
        topk_ids = inputs["topk_ids"]
        topk_probs = inputs["topk_probs"]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits
        total_loss = logits.new_tensor(0.0)

        if self.kl_weight > 0.0:
            if self.alm:
                kl = sparse_kl_loss_from_topk_alm(logits, topk_ids, topk_probs, loss_mask, labels)
            else:
                kl = sparse_kl_loss_from_topk(logits, topk_ids, topk_probs, loss_mask)
            total_loss = total_loss + self.kl_weight * kl

        if self.ce_weight > 0.0:
            ce_mask = labels != self.ignore_index
            if ce_mask.any():
                ce = F.cross_entropy(
                    logits[ce_mask],
                    labels[ce_mask],
                    ignore_index=self.ignore_index,
                )
                total_loss = total_loss + self.ce_weight * ce

        return (total_loss, outputs) if return_outputs else total_loss


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    ).train()

    if config.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def build_training_args(config):
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.nepochs,
        learning_rate=config.lr,
        bf16=config.bf16,
        logging_steps=1,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="epoch" if config.validation_ratio > 0.0 else "no",
        report_to="none",
        remove_unused_columns=False,
        lr_scheduler_type="constant",
        seed=config.seed,
    )


def main():
    config = parse_args()
    if config.debug:
        install_debug_hook()

    tokenizer = load_tokenizer(config.model_id)
    model = load_model(config)
    hf_dataset = load_dataset("json", data_files={"train": config.dataset_path})["train"]
    if config.validation_ratio > 0.0:
        split_dataset = hf_dataset.train_test_split(
            test_size=config.validation_ratio,
            seed=config.seed,
            shuffle=True,
        )
        train_hf_dataset = split_dataset["train"]
        eval_hf_dataset = split_dataset["test"]
    else:
        train_hf_dataset = hf_dataset
        eval_hf_dataset = None

    train_dataset = TopKDistilDataset(train_hf_dataset)
    eval_dataset = TopKDistilDataset(eval_hf_dataset) if eval_hf_dataset is not None else None
    collator = DistilCollator(pad_token_id=tokenizer.pad_token_id, max_length=config.max_length)

    print(f"Train on GSM8K: {len(train_dataset)} train rows")
    if eval_dataset is not None:
        print(f"Validate on GSM8K: {len(eval_dataset)} eval rows")
    trainer = TopKDistilTrainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        kl_weight=config.kl_weight,
        ce_weight=config.ce_weight,
        ignore_index=IGNORE_INDEX,
        alm=config.alm,
    )
    trainer.train()
    trainer.save_model(config.final_save_dir)
    print(f"Saved final model to {config.final_save_dir}")


if __name__ == "__main__":
    main()
