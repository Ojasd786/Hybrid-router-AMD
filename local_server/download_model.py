"""
download_model.py

Downloads the local Hugging Face model used for inference.
Run once before starting the local inference server.
"""

from huggingface_hub import snapshot_download

from local_server.config import MODEL_NAME


MODEL_DIR = "./models"


def download_model() -> None:
    """
    Download the configured Hugging Face model.
    """

    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\nModel downloaded successfully.\nLocation: {MODEL_DIR}")

    except Exception as error:
        print(f"\nFailed to download model:\n{error}")
        raise


if __name__ == "__main__":
    print(f"Downloading model: {MODEL_NAME}\n")
    download_model()