import os

MODEL_NAME = os.getenv(
    "LOCAL_MODEL_NAME",
    "Qwen/Qwen2.5-3B-Instruct"
)

HOST = os.getenv(
    "HOST",
    "0.0.0.0"
)

PORT = int(
    os.getenv(
        "PORT",
        "8000"
    )
)

MAX_NEW_TOKENS = int(
    os.getenv(
        "MAX_NEW_TOKENS",
        "512"
    )
)

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.2"
    )
)

TOP_P = float(
    os.getenv(
        "TOP_P",
        "0.95"
    )
)

# Use CPU for local testing on Mac
DEVICE = os.getenv(
    "DEVICE",
    "cpu"
)

DTYPE = os.getenv(
    "DTYPE",
    "float32"
)