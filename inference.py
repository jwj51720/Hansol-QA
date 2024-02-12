import os
import torch
import json
import argparse  # argparse 모듈 임포트
from modules import *


def main(CFG):
    print("**START**")
    model = get_model(CFG, "inference")
    train_loader, valid_loader = get_loader(CFG)
    print("**LOAD DATA COMPLETE**")
    trainer(CFG, model, train_loader, valid_loader)
    print("**MODEL INFERENCE COMPLETE**")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Team 3 - Hansoldeco")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        default="./base_config.json",
        help="JSON Config File Path",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        required=True,
        default="2",
        help="GPU number you want to use",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    device = "cpu"
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        chosen_gpu = min(args.gpu, n_gpu - 1)
        device = f"cuda:{chosen_gpu}"
        print(f"CUDA is available. {n_gpu} GPU(s) detected. Using {device}.")
    else:
        print(f"CUDA is not available. Using {device}.")
    config["DEVICE"] = device
    if not os.path.exists(config['SAVE_PATH']):
        os.makedirs(config['SAVE_PATH'])
    main(config)
