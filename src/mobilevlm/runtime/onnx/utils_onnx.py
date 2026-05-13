import torch
import numpy as np

IMAGE_TOKEN_INDEX = -200

def build_prompt(question: str) -> str:
    conv = conv_vicuna_v1.copy()
    conv.append_message(conv.roles[0], "<image>" + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def tokenizer_image_token_onnx(prompt, tokenizer):
    prompt_chunks = [
        [tokenizer.bos_id()] + tokenizer.encode(chunk, out_type=int) 
        for chunk in prompt.split('<image>')
    ]
    # prompt_chunks = [[chunk1], [chunk2]]
    '''
    [
      [1, 319, 13563, ..., 29901, 29871],  # 1 is tokenizer.bos_token_id
      [1, 29871, 13, ..., 13566, 29901]
    ]
    '''
    
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_id():
        offset = 1
        input_ids.append(prompt_chunks[0][0])  # input_ids = [1]
    
    for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(x[offset:])  # input_ids = [1, 319, 13563, ..., -200, 29871, ..., 29901]

    return np.array(input_ids, dtype=np.int64)


def np_empty_kv(batch_size=1, dtype=np.float32):
    """
    Create empty KV cache for decoder (NumPy version, no torch).

    Args:
        model: loaded model (used only for config access)
        batch_size: batch size
        dtype: numpy dtype (e.g., np.float32)

    Returns:
        List of (k, v) tuples for each layer
    """

    # Number of transformer layers
    num_layers = 24


    num_kv_heads = 16

    # Head dimension
    head_dim = 128

    empty_kv = []

    for _ in range(num_layers):
        # Create empty key tensor: (B, num_kv_heads, 0, head_dim)
        k = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=dtype,
        )

        # Create empty value tensor: (B, num_kv_heads, 0, head_dim)
        v = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim),
            dtype=dtype,
        )

        empty_kv.append((k, v))

    return empty_kv

import dataclasses
from enum import auto, Enum
from typing import List


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)