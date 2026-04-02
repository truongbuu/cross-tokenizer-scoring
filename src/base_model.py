# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from abc import ABC
import torch
from transformers import AutoModelForCausalLM, Qwen2Tokenizer


class BaseModel(ABC):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: "cpu",
        vocab_file="Qwen2/vocab.json",
        merge_file="Qwen2/merges.txt"
    ):
        """
        Initialize the Base LLM (HuggingFace).

        Args:
            pretrained_model_name_or_path (str): Path to the HuggingFace model.
            device (str): device name (either "gpu" or "cpu").

        Attributes:
            device (str): Device to store LLM.
            llm (AutoModelForCausalLM): huggingface llm inferface.
            t_wrapper (BaseTokenizer): tokenizer wrapper for string processing.
        """

        self.device = device
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path
        ).to(self.device)

        self.orig_tokenizer = Qwen2Tokenizer(vocab_file, merge_file, fast_mode=False)

    @torch.inference_mode()
    def logprobs(
        self,
        input_str: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache=True,
        cache_enc=[],
    ):
        """
        Uses transformer API to compute the logprobs of all tokens for one
        step. Takes inputs either as string or token ids.
        """

        # input to the method needs to be either string or token ids
        assert (input_str, input_ids) != (None, None)
        
        if input_ids is None:
            input_ids = self.tokenizer.encode(input_str)
        input_len = len(input_ids)
        input_ids = torch.Tensor(input_ids).int().to(self.device)

        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, ...]

        assert len(input_ids.shape) == 2, "Incorrect input ids dimension"

        if not use_cache:
            outputs = self.llm(input_ids, use_cache=False)
        else:
            cache_position = torch.arange(
                len(cache_enc), input_len, device=self.llm.device
            )
            outputs = self.llm(
                input_ids[:, len(cache_enc) :],
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            )

        log_probs = outputs.logits.log_softmax(dim=-1)#[:, -1, :]

        return log_probs, outputs.past_key_values

if __name__ == "__main__":
    Qwen_model = BaseModel("Qwen/Qwen2-0.5B", device="cuda")
    Qwen_model.logprobs_next_token(input_str="Large language models are")
