from typing import Optional

import torch
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer

def build_mtp_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    draft_length: int = 4,
    mask_token_id: Optional[int] = None,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
):
    """
    Build MTP dataset from a HuggingFace dataset.
    """
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
        if mask_token_id == tokenizer.unk_token_id:
            raise ValueError("Mask token ID is not set")
    
    dataset = dataset.shuffle(seed=shuffle_seed)

    def insert_mask_tokens(input_token_ids, output_token_ids):
        input_seqlen = input_token_ids.shape[1]
        output_seqlen = output_token_ids.shape[1]

        # Position IDs
        position_ids = torch.arange(0, output_seqlen)[:, None] + torch.arange(0, draft_length+1)[None, :]
        position_ids = position_ids.reshape(1, -1)
        position_ids = position_ids.to(output_token_ids.device)

        # get position_ids for input tokens
        input_position_ids = torch.arange(0, input_seqlen)[None] # [1, input_seqlen]
        position_ids = torch.cat([input_position_ids, position_ids + input_seqlen], dim=-1) # [1, input_seqlen + output_seqlen + draft_length]

        # set labels
        labels = torch.cat([torch.full((1, input_seqlen), -100, dtype=output_token_ids.dtype), output_token_ids], dim=-1) # [1, seqlen]; [[-100, ..., -100 || output_token_ids]]
        labels = torch.cat([labels, torch.full((1, draft_length+1), -100, dtype=labels.dtype)], dim=-1) # [1, seqlen + draft_length]; [[-100, ..., -100 || output_token_ids || -100, ..., -100]]
        labels = torch.gather(labels, 1, position_ids+1) # [1, input_seqlen + output_seqlen + draft_length]
        
        # insert <mask> tokens to output_ids
        output_token_ids = output_token_ids.transpose(0, 1)
        output_token_ids = torch.cat([output_token_ids, torch.full((output_seqlen, draft_length), mask_token_id)], dim=-1)
        output_token_ids = output_token_ids.reshape(1, -1)

        # merge with input_token_ids
        input_ids = torch.cat([input_token_ids, output_token_ids], dim=-1)

        # set gate_mask (for gated LoRA computation)
        # and regular_token_mask (for latent consistency loss, accuracy, kv cache extraction)
        gate_mask = torch.where((input_ids == mask_token_id) & (labels != -100), 1, 0)
        regular_token_mask = torch.where((input_ids != mask_token_id), 1, 0)

        return input_ids, position_ids, gate_mask, regular_token_mask

    def tokenize_function(x):
        input_token_ids = tokenizer(x["input"], return_tensors="pt", return_attention_mask=False)["input_ids"]
        output_token_ids = tokenizer(x["output"], return_tensors="pt", return_attention_mask=False)["input_ids"]
        input_ids, position_ids, gate_mask, regular_token_mask = insert_mask_tokens(input_token_ids, output_token_ids)
        
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "gate_mask": gate_mask,
            "regular_token_mask": regular_token_mask,
            "total_len": int(input_ids.shape[1])
        }

    dataset = dataset.map(tokenize_function, num_proc=num_proc, remove_columns=dataset.column_names)
    dataset.set_format(type='torch', columns=["input_ids", "position_ids", "gate_mask", "regular_token_mask", "total_len"])
    return dataset