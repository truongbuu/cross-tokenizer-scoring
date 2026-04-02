# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from abc import ABC
import torch
from transformers import AutoModelForCausalLM, Qwen2Tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

class BaseModel(ABC):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: str = "cuda",
        vocab_file="Qwen2/vocab.json",
        merge_file="Qwen2/merges.txt"
    ):
        """
        Initialize the Base LLM (HuggingFace).
        Args:
            pretrained_model_name_or_path (str): Path to the HuggingFace model.
            device (str): device name (defaults to "cuda").
        """
        self.device = device

        # Force float32 for sparse operations
        self.sparse_dtype = torch.float32
        
        # Model can use bfloat16 if available
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32
            
        # Load model with proper device mapping
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            **model_kwargs
        )
        
        # Keep tokenizer on CPU
        self.orig_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            use_fast=False, 
            trust_remote_code=True
        )
        #Qwen2Tokenizer(vocab_file, merge_file, fast_mode=False)

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
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.to(self.llm.device)  
        
        input_len = len(input_ids)

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
