"""
inference.py

Runs local inference using the loaded Hugging Face model.
"""

import torch

from local_server.model_loader import load_model
from local_server.config import MAX_NEW_TOKENS
from local_server.config import TEMPERATURE
from local_server.config import TOP_P

tokenizer, model = load_model()


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from the local model.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            temperature=TEMPERATURE,

            top_p=TOP_P,

            do_sample=False

        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]

    answer = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True

    )

    return answer.strip()