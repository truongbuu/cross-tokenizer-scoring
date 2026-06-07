import sys
import argparse

sys.path.append("./src")

from src.subvocab_model import SubVocabPredLLM


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Local model path or Hugging Face model name. Defaults to Qwen/Qwen2-0.5B.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on, e.g. cuda or cpu.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Today's weather is",
        help="Input prompt.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of subtokens to generate.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    SubLLM = SubVocabPredLLM(
        args.model_path,
        device=args.device,
        sub_vocab_merge="./qwen_vocabs/subset_vocabs/subsetQwen2_10000/10000_vocab.json",
        sub_merges_file="./qwen_vocabs/subset_vocabs/subsetQwen2_10000/10000_merges.txt",
        vocab_file="./qwen_vocabs/orig_vocabs/Qwen2/vocab.json",
        merge_file="./qwen_vocabs/orig_vocabs/Qwen2/merges.txt",
    )

    input_prompt = args.prompt

    # sub_tokenizer is the tokenizer for the sub-vocab.
    sub_encs = SubLLM.sub_tokenizer.encode(input_prompt)
    sampler_state = None

    # Greedy decoding.
    for _ in range(args.max_new_tokens):
        preds, sampler_state = SubLLM.prob_next_subtoken(
            sub_encs,
            sampler_state=sampler_state,
        )

        if preds is None:
            print("Generation stopped: prob_next_subtoken returned None.")
            break

        next_token = preds.argmax().item()
        sub_encs.append(next_token)

        print(SubLLM.sub_tokenizer.decode(sub_encs))


if __name__ == "__main__":
    main()