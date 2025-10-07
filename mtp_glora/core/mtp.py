import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtp_glora.models import SamplerHead
from .loss import StableSoftCrossEntropy

class MTPModel(nn.Module):
     """
     MTP with Gated LoRA model.
     """

     def __init__(
          self,
          model,
          draft_length: int = 4,
          kernel_options: Optional[Dict] = None,
     ):
          super().__init__()
          self.model = model
          self.sampler_head = SamplerHead(model.config).to(model.device).to(model.dtype)
          self.draft_length = draft_length
          self.kernel_options = kernel_options

     def forward(
          self,
          input_ids: torch.Tensor,
          attention_mask: torch.Tensor,
          gate_mask: torch.Tensor,
          regular_token_mask: torch.Tensor,
          position_ids: Optional[torch.Tensor],
          past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]],
     ):
          """
          Returns:
               - 'loss': soft SCE mean (scalar)
               - 'correct': num correct sampler head preds [B, D]
               - 'num_regular_tokens': scalar long tensor (R)
               - 'past_key_values': passthrough from base model
          """
     
          outputs = self.model(
               input_ids=input_ids,
               attention_mask=attention_mask,
               gate_mask=gate_mask,
               position_ids=position_ids,
               past_key_values=past_key_values,
               use_cache=True,
               output_hidden_states=True,
               kernel_options=self.kernel_options,
          )

          logits: torch.Tensor = outputs.logits
          hidden_states: torch.Tensor = outputs.hidden_states[-1]
          B, S, V = logits.shape
          H = hidden_states.size(-1)
          D = int(self.draft_length)
          device = logits.device
          dtype = logits.dtype


          if B != 1:
               import warnings
               warnings.warn(f"forward expects batch size=1; got B={B}. Using sample-0 layout.")

          g0 = gate_mask[:, :, 0] if gate_mask.dim() == 3 else gate_mask
          b0 = 0
          block = D + 1

          first_ones = (g0[b0] == 1).nonzero(as_tuple=True)[0]
          assert first_ones.numel() > 0, "gate_mask must contain at least one '1'."
          first_mask_idx = int(first_ones[0].item())
          start_idx = first_mask_idx - 1

          last_regular_idx = int(regular_token_mask[b0].nonzero(as_tuple=True)[0][-1].item())
          valid_seqlen = last_regular_idx + block

          if valid_seqlen <= start_idx:
               zero = logits.new_zeros(())
               return {
                    'loss': zero,
                    'correct': torch.zeros(B, D, dtype=torch.long, device=device),
                    'num_regular_tokens': torch.tensor(0, device=device, dtype=torch.long),
                    'past_key_values': outputs.past_key_values,
               }
          
          R = (valid_seqlen - start_idx + block - 1) // block # number of regular tokens
          reg_slice = slice(start_idx, start_idx + block * R, block) # regular indices

          reg_logits = logits[:, reg_slice, :]
          next_tokens = reg_logits.argmax(dim=-1)

          embed_weight = self.model.get_input_embeddings().weight.detach()
          lm_head_weight = self.model.lm_head.weight.detach()

          max_d = min(D, max(R - 1, 0))
          correct = torch.zeros(B, D, dtype=torch.long, device=device)

          # Accumulate soft SCE as SUM, normalize by count at the end
          ce_sum = torch.zeros((), dtype=torch.float32, device=device)
          ce_cnt = torch.zeros((), dtype=torch.float32, device=device)

          for d in range(max_d):
               end = R - (d + 1)
               if end <= 0:
                    break

               # strided window positions for this step
               s0 = start_idx + (d + 1)
               s1 = s0 + block * end

               # teacher (base) features at draft positions
               draft_hidden = hidden_states[:, s0:s1:block, :]

               # student (sampler) inputs
               prev_tok = next_tokens[:, :end]
               prev_emb = F.embedding(prev_tok, embed_weight)
               sampler_in = torch.cat([prev_emb, draft_hidden], dim=-1)
               sampler_hidden = self.sampler_head(sampler_in)
               sampler_logits = F.linear(sampler_hidden, lm_head_weight)

               teacher_logits_step = reg_logits[:, (d + 1):(d + 1 + end), :]
               step_loss_sum = StableSoftCrossEntropy.apply(sampler_logits, teacher_logits_step)
               ce_sum += step_loss_sum
               ce_cnt += end * B

               with torch.no_grad():
                    samp_tgt = teacher_logits_step.argmax(dim=-1)
                    samp_pred = sampler_logits.argmax(dim=-1)
                    correct[:, d] = (samp_pred.eq(samp_tgt)).sum(dim=1)

                    next_tokens[:, :end] = samp_pred

          loss = (ce_sum / ce_cnt.clamp_min(1.0)).to(dtype)
          num_regular_tokens = torch.tensor(R, device=device, dtype=torch.long)

          return {
               'loss': loss,
               'correct': correct,
               'num_regular_tokens': num_regular_tokens,
               'past_key_values': outputs.past_key_values,
          }