# Cross-Tokenizer Scoring

Utilities for generating cross-tokenizer distillation data between Qwen subtoken scores and target-tokenized outputs.

## Scoring Gemma2 Tokenizer using Qwen2.5 on GSM8K

Run from the repository root (there are 80 splits in total):

```bash
python datagen/gsm8k_gemma.py \
  --qwen_dir ./Qwen2.5-0.5B-Instruct/ \
  --gemma_dir google/gemma-2-2b-it \
  --split 0
```

The script expects the split file at:

```text
./splits_jsonl_gsm8k_80/gsm8k_main_train_part_<split>.jsonl
```

It writes successful rows to:

```text
./out_jsonl_gsm8k_gemma/gsm8k_main_train_part_<split>.jsonl
```

Rows that cannot be processed are written to (should not happen):

```text
./out_jsonl_gsm8k_gemma/error_gsm8k_main_train_part_<split>.jsonl
```

## Training on GSM8K

Run training from the repository root with:

```bash
bash run_training.sh
```

This script launches `gemma_distillation.py` on the GSM8K training data at:

```text
./out_jsonl_gsm8k_gemma/gsm8k_gemma_filtered.jsonl
```

Notes:

- `--qwen_dir` must point to a Qwen model directory containing `vocab.json` and `merges.txt`.
- `--gemma_dir` can be a local model path or a Hugging Face model ID.
- The script loads both models on CUDA and uses a Gemma cache length of 2048.

## Decoding with a Qwen2 Sub-Vocabulary

`run_decode.py` runs greedy decoding with `src/subvocab_model.py` and the bundled Qwen2 sub-vocabulary files.

Run from the repository root:

```bash
python run_decode.py \
  --model_path Qwen/Qwen2-0.5B \
  --device cuda \
  --prompt "Today's weather is" \
  --max_new_tokens 256
```

Arguments:

- `--model_path`: local model path or Hugging Face model ID. Defaults to `Qwen/Qwen2-0.5B`.
- `--device`: device for inference, such as `cuda` or `cpu`. Defaults to `cuda`.
- `--prompt`: input prompt to decode from.
- `--max_new_tokens`: maximum number of subtokens to generate.

The script uses these vocabulary files:

```text
./qwen_vocabs/subset_vocabs/subsetQwen2_10000/10000_vocab.json
./qwen_vocabs/subset_vocabs/subsetQwen2_10000/10000_merges.txt
./qwen_vocabs/orig_vocabs/Qwen2/vocab.json
./qwen_vocabs/orig_vocabs/Qwen2/merges.txt
```

It prints the decoded text after each generated subtoken.
