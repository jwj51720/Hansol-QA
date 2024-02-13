import os
import torch
import json
import argparse  # argparse 모듈 임포트
from modules import *
import warnings


def main(CFG):
    print("**START**")
    model = get_model(CFG)
    train_loader, valid_loader = get_loader(CFG)
    print("**LOAD DATA COMPLETE**")
    training(CFG, model, train_loader, valid_loader)
    print("**MODEL TRAIN COMPLETE**")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Team 3 - Hansoldeco")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./base_config",
        help="Encrypted Config File Path",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=2,
        help="GPU number you want to use",
    )
    args = parser.parse_args()
    config = crypto_decode(args.config)
    device = "cpu"
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        chosen_gpu = min(args.gpu, n_gpu - 1)
        device = f"cuda:{chosen_gpu}"
        print(f"CUDA is available. {n_gpu} GPU(s) detected. Using {device}.")
    else:
        print(f"CUDA is not available. Using {device}.")
    config["DEVICE"] = device

    if not os.path.exists(config["SAVE_PATH"]):
        os.makedirs(config["SAVE_PATH"])
    config["START_TIME"] = start_time()

    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*TypedStorage is deprecated.*"
    )
    seed_everything(config["SEED"])
    main(config)
