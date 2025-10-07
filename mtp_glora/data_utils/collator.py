import warnings
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    and_masks,
    or_masks,
)


class MTPChunkedDataCollator:
    """
    Data collator for MTP training with mask token insertion.
    
    This collator:
        1. Support chunking for memory efficiency
        2. Creates appropriate attention masks for each chunk
        3. Support batching and padding
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        draft_length: int = 4,          # Number of mask tokens to insert after each output token
        chunk_size: int = 2048,         # Size of each chunk
        min_chunk_size: int = 1024,     # Minimum size of each chunk
        mask_token_id: Optional[int] = None,  # ID for mask token
        pad_token_id: Optional[int] = None,   # ID for padding token
        last_chunk_buckets: Optional[List[int]] = None,  # e.g., [1024, 2048, 3072, 4096, 5120, 6144]
    ):
        if chunk_size < draft_length + 1:
            raise ValueError(f"chunk_size ({chunk_size}) must be greater than draft_length ({draft_length})")
        if chunk_size < min_chunk_size:
            raise ValueError(f"chunk_size ({chunk_size}) must be greater than min_chunk_size ({min_chunk_size})")
        if draft_length < 0:
            raise ValueError(f"draft_length ({draft_length}) must be non-negative")

        self.tokenizer = tokenizer
        self.draft_length = draft_length
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size

        # buckets for last chunk padding
        if last_chunk_buckets is None:
            last_chunk_buckets = list(range(1024, chunk_size + 1, 1024))
            if not chunk_size in last_chunk_buckets:
                # when the chunk_size is not divisible by 1024, add the chunk_size to the last_chunk_buckets
                last_chunk_buckets.append(chunk_size)
        self.last_chunk_buckets = sorted(set(last_chunk_buckets))

        # Set mask token ID
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            self.mask_token_id = tokenizer.convert_tokens_to_ids('<mask>')
            if self.mask_token_id == tokenizer.unk_token_id:
                raise ValueError("Mask token not found in tokenizer. Please specify mask_token_id.")
        
        # Set padding token ID
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            raise ValueError("Padding token not found in tokenizer. Please specify pad_token_id.")

    def _pick_last_bucket(self, content_len: int) -> int:
        """Pick the smallest bucket >= max(content_len, min_chunk_size)."""
        need = max(content_len, self.min_chunk_size)
        for b in self.last_chunk_buckets:
            if b >= need:
                return b
        # Fallback: if content exceeds largest bucket (shouldn't happen if chunk_size <= max bucket)
        warnings.warn(
            f"[MTPChunkedDataCollator] content_len {content_len} exceeds largest bucket "
            f"{self.last_chunk_buckets[-1]}; falling back to {need}."
        )
        return need

    def padding_tensor(self, intensors: torch.Tensor, max_length: int, pad_value: int = 0) -> torch.Tensor:
        """
        Pad a tensor to the max_length.
        """
        bsz, seqlen = intensors.shape
        if max_length - seqlen <= 0:
            return intensors

        # Keep device & dtype consistent with source tensor
        padding_tensor = torch.full(
            (bsz, max_length - seqlen),
            pad_value,
            dtype=intensors.dtype,
            device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=-1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collate a batch of features with MTP processing.
        Split into rectified chunks which guarantee the 'regular token + mask token * draft_length' structure.

        Note:
            This function assumes that the batch_size is 1 for simplicity.
            Also, implicitly assumes that the input_seqlen (prompt length) is less than the chunk size.

        Args:
            features: List of dictionaries containing:
                - input_ids: Input token IDs
                - position_ids: Position IDs
                - gate_mask: Gate mask
                - regular_token_mask: Regular token mask
                
        Returns:
            List[Dict[str, Tensor]]: processed chunks ready for training
        """
        if len(features) > 1:
            warnings.warn(f"`MTPChunkedDataCollator` expects batch size to be 1, but got batch size: {len(features)}")
        
        feature = features[0]
        gate_mask = feature["gate_mask"]
        total_len = feature["input_ids"].shape[1]
        block_len = 1 + self.draft_length

        # Calculate chunk boundaries (split points)
        mask_indices = (gate_mask == 1).nonzero(as_tuple=True)[1]
        prompt_len = mask_indices[0].item() - 1 if len(mask_indices) > 0 else total_len

        split_points = [0]
        cursor = 0
        while cursor < total_len:
            is_first_chunk = (cursor == 0)
            base_capacity = self.chunk_size - prompt_len if is_first_chunk else self.chunk_size
            num_blocks = max(0, base_capacity // block_len)
            content_len = (prompt_len if is_first_chunk else 0) + num_blocks * block_len

            cursor += content_len
            split_points.append(min(cursor, total_len))

        # Remove duplicate boundaries
        if len(split_points) > 1 and split_points[-1] == split_points[-2]:
            split_points.pop()

        # Define padding values for each tensor type
        keys_to_process = {
            "input_ids": self.pad_token_id,
            "gate_mask": -1,
            "position_ids": 0,
            "regular_token_mask": 0,
        }

        # Create chunks and pad based on the calculated boundaries
        batch_chunks = []
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            is_last_chunk = (i == len(split_points) - 2)

            content_len = end - start
            if is_last_chunk:
                target_len = self._pick_last_bucket(content_len)
            else:
                target_len = self.chunk_size

            batch_chunks.append({
                key: self.padding_tensor(feature[key][:, start:end], target_len, pad_val)
                for key, pad_val in keys_to_process.items()
            })

        return batch_chunks


class StreamingKVCacheManager:
    """
    KV cache manager for streaming/sequential chunk processing.
    """
    

    def __init__(self):
        self.past_seen_tokens = 0


    def reset_cache(self):
        """Reset KV cache for new sequence."""
        self.past_seen_tokens = 0
    

    def prepare_data_with_kv_cache(
        self,
        chunk_data: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Prepare a chunk data for training with KV cache from previous chunks.
        Mainly, it creates block mask for flex attention in the current chunk.
        
        Args:
            chunk_data: Data for current chunk
            
        Returns:
            Processed chunk data with KV cache information
        """
        # Create block mask for flex attention in the current chunk
        chunk_data['attention_mask'] = self._create_block_mask_for_chunk(
            chunk_data,
        ).to(chunk_data['input_ids'].device)

        # Convert -1 to 0 for padding tokens to pass the gated LoRA computation for padding tokens
        # and unsqueeze(-1) for the broadcasting over hidden_size dimension
        chunk_data['gate_mask'] = torch.clamp(chunk_data['gate_mask'], min=0)
        chunk_data['gate_mask'] = chunk_data['gate_mask'][:, :, None]
        
        return chunk_data
    

    def extract_regular_kv_cache_for_next_chunk(
        self,
        chunk_data: Dict[str, torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
    ) -> Optional[DynamicCache]:
        """
        Create a new DynamicCache object with filtered tensors.

        Note:
            This function assumes that the batch_size is 1 for simplicity.
            If you want to use this function for a batch of size > 1, you need to modify the function.
            This is because increasing the batch size results in shorter chunk size, which offers no benefit.

        Args:
            chunk_data: Data for current chunk
            past_key_values: KV cache after forward pass for current chunk
            sanity_check: Whether to perform sanity check
        
        Returns:
            DynamicCache object with filtered tensors
        """
        bsz, _ = chunk_data['input_ids'].shape
        if bsz > 1:
            warnings.warn(f"`extract_regular_kv_cache_for_next_chunk` expects batch size to be 1, but got batch size: {bsz}")

        if past_key_values is None:
            return None

        # Extract regular token indices in the current chunk
        current_chunk_regular_indices = chunk_data['regular_token_mask'][0].nonzero(as_tuple=True)[0]
        num_new_regular_tokens = len(current_chunk_regular_indices)

        # Sanity Check
        if 'position_ids' in chunk_data:
            device = chunk_data['position_ids'].device
            extracted_pos_ids = chunk_data['position_ids'][:, current_chunk_regular_indices]
            
            # Expected position_ids are [past_seen, past_seen+1, ...]
            expected_pos_ids = torch.arange(
                self.past_seen_tokens,
                self.past_seen_tokens + num_new_regular_tokens, 
                device=device
            ).expand_as(extracted_pos_ids)
            
            if not torch.equal(extracted_pos_ids, expected_pos_ids):
                print(f"extract_regular_kv_cache_for_next_chunk: Sanity check failed. Position IDs mismatch. Expected: {expected_pos_ids}, Actual: {extracted_pos_ids}")
                print(f"self.past_seen_tokens: {self.past_seen_tokens}, num_new_regular_tokens: {num_new_regular_tokens}")
                print(f"Given past_key_values length: {past_key_values[0][0].shape[2]}")

        # Split, filter, and concatenate
        filtered_kv_tuples = []
        current_chunk_size = chunk_data['input_ids'].shape[1]
        for key, value in past_key_values:
            past_key, current_key = key.split([self.past_seen_tokens, current_chunk_size], dim=2)
            past_value, current_value = value.split([self.past_seen_tokens, current_chunk_size], dim=2)

            new_regular_key = current_key[:, :, current_chunk_regular_indices, :]
            new_regular_value = current_value[:, :, current_chunk_regular_indices, :]

            filtered_key = torch.cat([past_key, new_regular_key], dim=2).detach()
            filtered_value = torch.cat([past_value, new_regular_value], dim=2).detach()

            filtered_kv_tuples.append((filtered_key, filtered_value))

        # Update the number of accumulated regular tokens
        self.past_seen_tokens += num_new_regular_tokens
        
        return DynamicCache.from_legacy_cache(past_key_values=tuple(filtered_kv_tuples))


    def _create_block_mask_for_chunk(
        self,
        chunk_data: Dict[str, torch.Tensor],
    ) -> BlockMask:
        """
        Create a block mask for flex attention in the current chunk.

        Note:
            This function assumes that the batch_size is 1 for simplicity.

        Args:
            chunk_data: Data for current chunk

        Returns:
            Block mask for flex attention in the current chunk
        """

        bsz, seqlen = chunk_data['input_ids'].shape
        if bsz > 1:
            warnings.warn(f"`_create_block_mask_for_chunk` expects batch size to be 1, but got batch size: {bsz}")

        past_seen_tokens = self.past_seen_tokens
        if past_seen_tokens > 0:
            gate_mask_with_past_kv = torch.cat([
                torch.zeros(bsz, past_seen_tokens, dtype=chunk_data['gate_mask'].dtype, device=chunk_data['gate_mask'].device),
                chunk_data['gate_mask']
            ], dim=1)
        else:
            gate_mask_with_past_kv = chunk_data['gate_mask']
        
        num_prefix_x = (gate_mask_with_past_kv == 0).long().cumsum(dim=1)

        # Predicates
        def is_causal(b, h, q_idx, kv_idx): return q_idx+past_seen_tokens >= kv_idx
        def is_query_not_pad(b, h, q_idx, kv_idx): return gate_mask_with_past_kv[b, q_idx+past_seen_tokens] != -1
        def is_key_not_pad(b, h, q_idx, kv_idx): return gate_mask_with_past_kv[b, kv_idx] != -1
        
        # Query token type
        def is_query_x(b, h, q_idx, kv_idx): return gate_mask_with_past_kv[b, q_idx+past_seen_tokens] == 0
        def is_query_m(b, h, q_idx, kv_idx): return gate_mask_with_past_kv[b, q_idx+past_seen_tokens] == 1
        
        # Key token type
        def is_key_x(b, h, q_idx, kv_idx): return gate_mask_with_past_kv[b, kv_idx] == 0
        def is_key_m(b, h, q_idx, kv_idx): return gate_mask_with_past_kv[b, kv_idx] == 1
        
        # 'm' block
        def is_same_m_block(b, h, q_idx, kv_idx): return num_prefix_x[b, q_idx+past_seen_tokens] == num_prefix_x[b, kv_idx]

        # 'x' query rule: query is 'x' AND key is 'x'
        x_rule = and_masks(is_query_x, is_key_x)
        
        # 'm' query rule: query is 'm' AND (key is 'x' OR (key is 'm' AND same block))
        m_rule = and_masks(
            is_query_m,
            or_masks(
                is_key_x,
                and_masks(is_key_m, is_same_m_block)
            )
        )
        
        mask_func = and_masks(
            is_causal,
            is_query_not_pad,
            is_key_not_pad,
            or_masks(x_rule, m_rule)
        )
        
        return create_block_mask(
            mask_mod=mask_func,
            B=None,
            H=None,
            Q_LEN=seqlen,
            KV_LEN=self.past_seen_tokens+seqlen,
        )
