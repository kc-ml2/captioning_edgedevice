#!/usr/bin/env python3
"""Run deterministic MobileVLM captioning directly from converted weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, LlamaTokenizer

HEADS, HEAD_DIM, HIDDEN, LAYERS = 16, 128, 2048, 24
IMAGE_TOKEN = -200


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--question", default="What objects are visible in the scene?")
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def dequantize_affine_q4(weight, scales, biases, group_size=64):
    """Independently unpack MLX affine q4 weights into float32 PyTorch tensors."""
    packed = weight.to(torch.int64)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    values = ((packed.unsqueeze(-1) >> shifts) & 0xF).flatten(-2).float()
    groups = values.shape[-1] // group_size
    shape = values.shape[:-1] + (groups, group_size)
    return (values.reshape(shape) * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).flatten(-2)


def load_weights(path: Path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    stored = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(path / shard, framework="pt", device="cpu") as f:
            stored.update({name: f.get_tensor(name) for name in f.keys()})

    result = {}
    for name, tensor in stored.items():
        if name.endswith((".scales", ".biases")):
            continue
        prefix = name.removesuffix(".weight")
        scales = stored.get(f"{prefix}.scales")
        biases = stored.get(f"{prefix}.biases")
        result[name] = (
            dequantize_affine_q4(tensor, scales, biases)
            if scales is not None and biases is not None
            else tensor.float()
        )
    return result


def linear(x, w): return F.linear(x, w)
def norm(x, w): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w

def layer_norm(x, w, b): return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)

def rope(x, positions):
    dims = torch.arange(0, x.shape[-1], 2, dtype=torch.float32)
    inv = 1 / (10000 ** (dims / x.shape[-1]))
    f = torch.outer(positions.float(), inv)
    e = torch.cat((f, f), -1)[None, None]
    a, b = x.chunk(2, -1)
    return x * e.cos() + torch.cat((-b, a), -1) * e.sin()


def vision(pixels, w):
    r = "vision_model.vision_model"
    patch = F.conv2d(pixels.permute(0,3,1,2), w[f"{r}.embeddings.patch_embedding.weight"].permute(0,3,1,2), stride=14)
    patch = patch.flatten(2).transpose(1,2)
    x = torch.cat((w[f"{r}.embeddings.class_embedding"].reshape(1,1,1024), patch), 1)
    x = x + w[f"{r}.embeddings.position_embedding.weight"]
    x = layer_norm(x, w[f"{r}.pre_layrnorm.weight"], w[f"{r}.pre_layrnorm.bias"])
    for n in range(23):
        p = f"{r}.encoder.layers.{n}"
        y = layer_norm(x, w[f"{p}.layer_norm1.weight"], w[f"{p}.layer_norm1.bias"])
        q = linear(y,w[f"{p}.self_attn.q_proj.weight"])+w[f"{p}.self_attn.q_proj.bias"]
        k = linear(y,w[f"{p}.self_attn.k_proj.weight"])+w[f"{p}.self_attn.k_proj.bias"]
        v = linear(y,w[f"{p}.self_attn.v_proj.weight"])+w[f"{p}.self_attn.v_proj.bias"]
        q=q.reshape(1,577,16,64).transpose(1,2); k=k.reshape(1,577,16,64).transpose(1,2); v=v.reshape(1,577,16,64).transpose(1,2)
        a=F.softmax(q@k.transpose(-1,-2)/8,-1)@v
        a=a.transpose(1,2).reshape(1,577,1024)
        x=x+linear(a,w[f"{p}.self_attn.out_proj.weight"])+w[f"{p}.self_attn.out_proj.bias"]
        y=layer_norm(x,w[f"{p}.layer_norm2.weight"],w[f"{p}.layer_norm2.bias"])
        y=linear(y,w[f"{p}.mlp.fc1.weight"])+w[f"{p}.mlp.fc1.bias"]
        y=y*torch.sigmoid(1.702*y)
        x=x+linear(y,w[f"{p}.mlp.fc2.weight"])+w[f"{p}.mlp.fc2.bias"]
    return x[:,1:]


def projector(x,w):
    x=linear(x,w['projector.mlp.mlp.0.weight'])+w['projector.mlp.mlp.0.bias']; x=F.gelu(x)
    x=linear(x,w['projector.mlp.mlp.2.weight'])+w['projector.mlp.mlp.2.bias']
    x=F.avg_pool2d(x.transpose(1,2).reshape(1,2048,24,24),2,2)
    c=F.conv2d(x,w['projector.peg.peg.0.weight'].permute(0,3,1,2),w['projector.peg.peg.0.bias'],padding=1,groups=2048)
    return (x+c).flatten(2).transpose(1,2)


def llm_layer(x,w,n,positions,past=None):
    p=f'language_model.layers.{n}'; y=norm(x,w[f'{p}.input_layernorm.weight']); length=x.shape[1]
    q=linear(y,w[f'{p}.self_attn.q_proj.weight']).reshape(1,length,HEADS,HEAD_DIM).transpose(1,2)
    k=linear(y,w[f'{p}.self_attn.k_proj.weight']).reshape(1,length,HEADS,HEAD_DIM).transpose(1,2)
    v=linear(y,w[f'{p}.self_attn.v_proj.weight']).reshape(1,length,HEADS,HEAD_DIM).transpose(1,2)
    q=rope(q,positions); k=rope(k,positions)
    if past: k=torch.cat((past[0],k),2); v=torch.cat((past[1],v),2); mask=0
    else: mask=torch.triu(torch.full((length,length),float('-inf')),1)
    a=F.softmax(q@k.transpose(-1,-2)/math.sqrt(HEAD_DIM)+mask,-1)@v
    x=x+linear(a.transpose(1,2).reshape(1,length,HIDDEN),w[f'{p}.self_attn.o_proj.weight'])
    y=norm(x,w[f'{p}.post_attention_layernorm.weight'])
    y=F.silu(linear(y,w[f'{p}.mlp.gate_proj.weight']))*linear(y,w[f'{p}.mlp.up_proj.weight'])
    return x+linear(y,w[f'{p}.mlp.down_proj.weight']),(k,v)


def expand_square(image, color):
    if image.width==image.height:return image
    s=max(image.size); result=Image.new('RGB',(s,s),color); result.paste(image,((s-image.width)//2,(s-image.height)//2)); return result


def main():
    a=args(); w=load_weights(a.model)
    processor=CLIPImageProcessor.from_pretrained(a.model)
    image=Image.open(a.image).convert('RGB'); color=tuple(int(x*255) for x in processor.image_mean)
    pixels=processor.preprocess(expand_square(image,color),return_tensors='pt')['pixel_values'].permute(0,2,3,1).float()
    tokenizer=LlamaTokenizer.from_pretrained(a.model,use_fast=False)
    prompt=("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n"+a.question+" ASSISTANT:")
    chunks=[tokenizer(c).input_ids for c in prompt.split('<image>')]
    ids=chunks[0]+[IMAGE_TOKEN]+chunks[1][1:]; pos=ids.index(IMAGE_TOKEN)
    with torch.inference_mode():
        image_features=projector(vision(pixels,w),w)
        embed=w['language_model.embed_tokens.weight']
        x=torch.cat((embed[torch.tensor(ids[:pos])],image_features[0],embed[torch.tensor(ids[pos+1:])]),0).unsqueeze(0)
        caches=[]
        for n in range(LAYERS): x,c=llm_layer(x,w,n,torch.arange(x.shape[1])); caches.append(c)
        generated=[]
        for step in range(a.max_tokens):
            token=(linear(norm(x,w['language_model.norm.weight']),w['lm_head.weight'])[:,-1]).argmax(-1)
            generated.append(token.item())
            if token.item()==2: break
            x=embed[token].unsqueeze(1)
            for n in range(LAYERS): x,c=llm_layer(x,w,n,torch.tensor([caches[n][0].shape[2]]),caches[n]); caches[n]=c
    caption=tokenizer.decode(generated,skip_special_tokens=True).strip()
    result={'prompt':prompt,'input_ids':ids,'pixel_shape':list(pixels.shape),'multimodal_length':len(ids)-1+144,'generated_tokens':generated,'caption':caption}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
