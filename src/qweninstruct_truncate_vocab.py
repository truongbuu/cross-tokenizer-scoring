import json
import argparse
import copy

form1 = {
"lstrip": False,
"normalized": False,
"rstrip": False,
"single_word": False,
"special": True
}

form2 = {
"lstrip": False,
"normalized": False,
"rstrip": False,
"single_word": False,
"special": False
}

form1_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]
form2_tokens = ["<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"]

to_added_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"]
config_json={
    "add_bos_token": False,
    "add_prefix_space": False,
    "additional_special_tokens": [
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>"
  ],
  "bos_token": None,
  "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
  "clean_up_tokenization_spaces": False,
  "eos_token": "<|im_end|>",
  "errors": "replace",
  "model_max_length": 131072,
  "pad_token": "<|endoftext|>",
  "split_special_tokens": False,
  "tokenizer_class": "Qwen2Tokenizer",
  "unk_token": None
}


def extract_and_save_merges(merge_file_path, output_file_path, n):
    merges = []
    header = None
    if n > 0:
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
    return merges


def extract_first_vocab_entries(vocab_path, output_path, n):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    sorted_vocab_items = sorted(vocab.items(), key=lambda x: x[1])
    selected_vocab_items = sorted_vocab_items[:256 + n]
    selected_vocab = {token: index for token, index in selected_vocab_items}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_vocab, f, ensure_ascii=False, indent=2)

    print(f"Saved first {256 + n} vocab entries to {output_path}")

def extract_first_vocabinstruct_entries(vocab_path, output_path, n):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    sorted_vocab_items = sorted(vocab.items(), key=lambda x: x[1])
    selected_vocab_items = sorted_vocab_items[:256 + n]
    selected_vocab = {token: index for token, index in selected_vocab_items}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_vocab, f, ensure_ascii=False, indent=2)

    print(f"Saved first {256 + n} vocab entries to {output_path}")
    return selected_vocab

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
    parser.add_argument("--tokenizer_json", type=str, default="tokenizer.json", help="Path to merges.txt file.")
    parser.add_argument("--out_vocab", type=str, default="vocab_trimmed.json", help="Output path for trimmed vocab.")
    parser.add_argument("--out_merges", type=str, default="merges_trimmed.txt", help="Output path for trimmed merges.")
    parser.add_argument("--out_config", type=str, default="tokenizer_config_trimmed.json", help="Path to merges.txt file.")
    parser.add_argument("--out_tokenizer_json", type=str, default="tokenizer_trimmed.json", help="Path to merges.txt file.")
    parser.add_argument("--verify", default=True, help="Verify trimmed merges against trimmed vocab.")

    args = parser.parse_args()

    new_merges = extract_and_save_merges(args.merges, args.out_merges, args.n)
    new_vocab = extract_first_vocabinstruct_entries(args.vocab, args.out_vocab, args.n)

    if args.verify:
        verify_merges_vs_vocab(args.out_merges, args.out_vocab)

    # Update tokenizer_config.json
    config_json["added_tokens_decoder"] = {}
    inv_vocab = load_vocab(args.out_vocab)

    added_tokens_id = {}
    for i in range(len(form1_tokens)):
        next_id = len(inv_vocab) + i
        config_json["added_tokens_decoder"][str(next_id)] = copy.deepcopy(form1)
        config_json["added_tokens_decoder"][str(next_id)]["content"] = form1_tokens[i]
        added_tokens_id[form1_tokens[i]] = next_id
    #import pdb; pdb.set_trace()
    for i in range(len(form2_tokens)):
        next_id = len(inv_vocab) + len(form1_tokens) + i
        config_json["added_tokens_decoder"][str(next_id)] = copy.deepcopy(form2)
        config_json["added_tokens_decoder"][str(next_id)]["content"] = form2_tokens[i]
        added_tokens_id[form2_tokens[i]] = next_id

    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump(config_json, f, ensure_ascii=False, indent=2)

    with open(args.tokenizer_json, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    tokenizer_json['model']['vocab'] = new_vocab
    tokenizer_json['model']['merges'] = new_merges

    for i in range(len(tokenizer_json['added_tokens'])):
        content = tokenizer_json['added_tokens'][i]['content']
        tokenizer_json['added_tokens'][i]['id'] = added_tokens_id[content]

    with open(args.out_tokenizer_json, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    #import pdb; pdb.set_trace()
