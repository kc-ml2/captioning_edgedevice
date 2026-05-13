import torch
import torch.nn as nn
from abc import ABC, abstractmethod
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None
from .vision_encoder import build_vision_tower
from .vision_projector import build_vision_projector
from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, \
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# Initializes and manages the vision encoder and projector to produce image embeddings for the LLM.
class MobileVLMMetaModel:

    def __init__(self, config):
        super(MobileVLMMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):  
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.mm_vision_tower = model_args.vision_tower
        vision_tower = model_args.vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        elif self.get_vision_tower().vision_tower_name != vision_tower:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
                vision_tower.load_model()
            else:
                vision_tower = self.vision_tower
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        # Build VisionTower
        vision_tower = build_vision_tower(model_args)
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower
        self.config.mm_hidden_size = vision_tower.hidden_size

        # Build Vision-Projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

        # In case it is frozen by LoRA
        for p in self.mm_projector.parameters():
            p.requires_grad = True
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


# Connects the vision encoder and LLM for multimodal processing.
class MobileVLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    # Prepare multi-modal inputs by inserting image features at <image> token positions.
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids,           # text prompt token_id from tokenizer: [1, 53]
        attention_mask, 
        past_key_values, 
        labels, 
        images=None,         # image feature from PyTorch
        image_features=None  # image feature from ONNX
    ):
        vision_tower = self.get_vision_tower()
        # prefill: seq_len > 1, decoding: seq_len == 1
        is_decoding_step = input_ids.shape[1] == 1

        # Skip multimodal input construction processing (prefill) -> text-only LLM processing
        if (
            (vision_tower is None) or                         # Text-only model
            (images is None and image_features is None) or    # Text-only input
            (is_decoding_step)                                # Multi-modal model + decoding step
        ):
            # Adjust attention_mask to match KV cache length during decoding
            if (
                (past_key_values is not None) and                       # KV cache available (used during decoding)
                (vision_tower is not None) and                          # multimodal model
                (images is not None or image_features is not None) and  # image input present
                (is_decoding_step)                                      # decoding step
            ):
                # Match attention_mask to KV cache length (+ current token)
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),  # key/value: (B, num_heads, **seq_len**, head_dim)
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
            return input_ids, attention_mask, past_key_values, None, labels
        
        # Use ONNX image_features if provided; otherwise compute with PyTorch.
        if image_features is None:
            if type(images) is list or images.ndim == 5:
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                image_features = [x.flatten(0, 1) for x in image_features]
            else:
                image_features = self.encode_images(images)

        else:
            if isinstance(image_features, list):
                image_features = [x.to(device=input_ids.device, dtype=self.get_model().embed_tokens.weight.dtype) for x in image_features]
            else:
                image_features = image_features.to(device=input_ids.device, dtype=self.get_model().embed_tokens.weight.dtype)


        # Multimodal input for LLM prefill step
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # multimodal LLM, but the current sample is not multimodal
            # FIXME: this is a hacky fix, for deepspeed zero3 to work
            # PASS
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]  # tensor([35])
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            # Replace each <image> token with image features
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]  # First image feature: [144, 2048]
                image_token_start = image_token_indices[0]          # tensor(35)

                # Use <im_start>/<im_end> tokens when training the multimodal adapter
                # else
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))  # [35, 2048]
                    cur_new_input_embeds.append(cur_image_features)                                                # [[35, 2048], [144, 2048]]
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                # Update indices of remaining <image> tokens after slicing
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            # cur_new_input_embeds: [[35, 2048], [144, 2048], [17, 2048]]
            cur_new_input_embeds = [x.to(device=input_ids.device) for x in cur_new_input_embeds]
            # cur_new_input_embeds: [196, 2048]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    # Suppress FutureWarning caused by Hugging Face Hub deprecating `resume_download`
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*`resume_download` is deprecated.*",
        category=FutureWarning,
    )

    is_cpu = (str(device) == "cpu")
    if is_cpu:
        device_map = None
    kwargs = {}
    if device_map is not None:
        kwargs["device_map"] = device_map

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        if is_cpu:
            raise ValueError("CPU inference does not support load_4bit in this setup. Use load_4bit=False.")
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float32 if is_cpu else torch.float16
    
    from transformers import LlamaTokenizer
    from .mobilellama import MobileLlamaForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    model = MobileLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    if is_cpu:
        model.to("cpu")

    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    vision_tower = model.get_vision_tower()
    if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()

    vt_dtype = torch.float32 if is_cpu else torch.float16
    vision_tower.to(device=device, dtype=vt_dtype)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len
