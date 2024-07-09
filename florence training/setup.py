import os
import subprocess

def setup_env():
    # Ensure GPU is available
    subprocess.run(["nvidia-smi"])

    # Install required packages
    packages = [
        "transformers",
        "flash_attn",
        "timm",
        "einops",
        "peft",
        "roboflow",
        "scikit-learn",
        "wandb",
        "git+https://github.com/roboflow/supervision.git"
    ]
    subprocess.run(["pip", "install", "-q"] + packages)

if __name__ == "__main__":
    setup_env()
    configure_api_keys()
