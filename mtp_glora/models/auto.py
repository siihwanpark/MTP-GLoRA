import os
from typing import Optional, Union

import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import (
    LlamaConfig,
    Qwen3Config,
)

from mtp_glora.utils import default_torch_dtype
from mtp_glora.models.modeling_qwen3 import Qwen3ForCausalLM
from mtp_glora.models.modeling_llama import LlamaForCausalLM


class AutoDistributedModelForCausalLM(AutoModelForCausalLMBase):
    """
    Base class for distributed causal language models.
    """
    _model_mapping = {
        Qwen3Config: [Qwen3ForCausalLM],
        LlamaConfig: [LlamaForCausalLM],
        # Add custom models here
    }


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **config_kwargs,
    ):
        """
        Load a distributed causal language model from a pretrained model name or path.

        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model.
            torch_dtype: The dtype of the model.
            device: The device of the model.
            cache_dir: The cache directory of the model.
            **config_kwargs: The kwargs for the config.
        """

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        assert (type(config) in cls._model_mapping), f"Unsupported config type: {type(config)}"
        model_cls = cls._model_mapping[type(config)][0]

        if device is None: device = torch.device("cpu")
        else: device = torch.device(device)
        if torch_dtype is None: torch_dtype = torch.get_default_dtype()

        with default_torch_dtype(torch_dtype), torch.device(device):
            model = model_cls(config)
        model.load_checkpoint(pretrained_model_name_or_path, cache_dir=cache_dir)

        return model