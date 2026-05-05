# src/utils/config.py
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

# Project root — absolute path, works from anywhere
PROJECT_ROOT = "/Users/khusbuagarwal/Desktop/clinical_trial_recruitment"


@dataclass
class Config:
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    hf_token:       str = os.getenv("HF_TOKEN", "")
    wandb_api_key:  str = os.getenv("WANDB_API_KEY", "")
    wandb_project:  str = os.getenv("WANDB_PROJECT", "clinical-trial-recruitment")

    # Model settings
    model_name:    str   = os.getenv("MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT")
    max_length:    int   = int(os.getenv("MAX_LENGTH",    "512"))
    batch_size:    int   = int(os.getenv("BATCH_SIZE",    "16"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    num_epochs:    int   = int(os.getenv("NUM_EPOCHS",    "5"))

    # Absolute paths — never relative
    data_dir:       str = os.path.join(PROJECT_ROOT, "data")
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    log_dir:        str = os.path.join(PROJECT_ROOT, "logs")

    @property
    def device(self) -> str:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def validate(self):
        issues = []
        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY missing")
        if not self.hf_token:
            issues.append("HF_TOKEN missing")
        if not self.wandb_api_key:
            issues.append("WANDB_API_KEY missing")
        return issues


cfg = Config()


if __name__ == "__main__":
    print("\n" + "="*45)
    print("   Clinical Trial Recruitment — Config")
    print("="*45)
    print(f"  Device      : {cfg.device}")
    print(f"  Data dir    : {cfg.data_dir}")
    print(f"  Model       : {cfg.model_name}")
    print(f"  Batch size  : {cfg.batch_size}")
    print(f"  Epochs      : {cfg.num_epochs}")
    print("="*45)
    issues = cfg.validate()
    if issues:
        print("⚠️  Missing keys:")
        for i in issues: print(f"   - {i}")
    else:
        print("✅ All keys loaded")
    print("="*45 + "\n")