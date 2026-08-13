from typing import List, Optional, Tuple, Union

import torch

from transformers import (
    AutoConfig,
    LlamaConfig,
    LlamaModel,
)

from transformers.modeling_outputs import BaseModelOutputWithPast


# ============================================================
# Config
# ============================================================

class MobileLlamaConfig(LlamaConfig):
    model_type = "mobilevlm"


# ============================================================
# Pure Backbone Model
# ============================================================

class MobileLlamaModel(LlamaModel):

    config_class = MobileLlamaConfig

    def __init__(self, config: LlamaConfig):

        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    
# ============================================================
# HuggingFace Registry
# ============================================================

AutoConfig.register(
    "mobilevlm",
    MobileLlamaConfig
)