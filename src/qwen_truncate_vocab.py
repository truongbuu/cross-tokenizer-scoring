import json
import argparse

def extract_and_save_merges(merge_file_path, output_file_path, n):
    merges = []
    header = None

    with open(merge_file_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and line.startswith("#version:"):
                header = line
                continue
            if not line:
                continue
            merges.append(line)
            if len(merges) >= n:
                break

    with open(output_file_path, "w", encoding="utf-8") as out:
        if header:
            out.write(header + "\n")
        for merge in merges:
            out.write(merge + "\n")

    print(f"Saved first {len(merges)} merges to {output_file_path}")


def extract_first_vocab_entries(vocab_path, output_path, n):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    sorted_vocab_items = sorted(vocab.items(), key=lambda x: x[1])
    selected_vocab_items = sorted_vocab_items[:256 + n]
    selected_vocab = {token: index for token, index in selected_vocab_items}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_vocab, f, ensure_ascii=False, indent=2)

    print(f"Saved first {256 + n} vocab entries to {output_path}")

def load_merges(merge_path):
    merges = []
    with open(merge_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 and line.startswith("#version:"):
                continue
            if not line:
                continue
            merges.append(tuple(line.split()))
    return merges


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    return inv_vocab


def verify_merges_vs_vocab(merges_path, vocab_path):
    merges = load_merges(merges_path)
    inv_vocab = load_vocab(vocab_path)

    success = True
    for i, (first, second) in enumerate(merges):
        vocab_index = 256 + i
        if vocab_index not in inv_vocab:
            print(f"Index {vocab_index} missing in vocab!")
            success = False
            continue

        vocab_token = inv_vocab[vocab_index]
        merged_candidate = first + second

        if vocab_token != merged_candidate:
            print(f"Mismatch at line {i}: vocab token '{vocab_token}' != merged pair '{merged_candidate}'")
            success = False

    if success:
        print("All merge entries correctly match vocab entries.")
    else:
        print("Some merge entries did NOT match vocab entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim vocab and merges files for a BPE tokenizer and verify.")
    parser.add_argument("--n", type=int, required=True, help="Number of merge rules to keep (vocab keeps 256 + n).")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocab.json file.")
    parser.add_argument("--merges", type=str, default="merges.txt", help="Path to merges.txt file.")
    parser.add_argument("--out_vocab", type=str, default="vocab_trimmed.json", help="Output path for trimmed vocab.")
    parser.add_argument("--out_merges", type=str, default="merges_trimmed.txt", help="Output path for trimmed merges.")
    parser.add_argument("--verify", default=True, help="Verify trimmed merges against trimmed vocab.")

    args = parser.parse_args()

    extract_and_save_merges(args.merges, args.out_merges, args.n)
    extract_first_vocab_entries(args.vocab, args.out_vocab, args.n)

    if args.verify:
        verify_merges_vs_vocab(args.out_merges, args.out_vocab)