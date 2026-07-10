"""
model_loader.py

Loads the Hugging Face tokenizer and language model.
"""

import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from local_server.config import MODEL_NAME
from local_server.config import DEVICE
from local_server.config import DTYPE


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model():
    """
    Load tokenizer and model.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE_MAP.get(DTYPE, torch.float16),
        device_map=DEVICE
    )

    model.eval()

    return tokenizer, model