import os
import torch
import json
import argparse  # argparse 모듈 임포트
from modules import *
import warnings
import wandb


def main(CFG):
    print("**START**")
    model = get_model(CFG)
    train_dataset, eval_dataset, _, _, = get_loader(CFG)
    print("**LOAD DATA COMPLETE**")
    trainer = HFTraining(CFG)
    trainer.run(model, train_dataset, eval_dataset)
    wandb.finish()
    print("**MODEL TRAIN COMPLETE**")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Team 3 - Hansoldeco")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Config File Path",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=2,
        help="GPU number you want to use",
    )
    args = parser.parse_args()
    # config = crypto_decode(args.config)
    with open(f'config/{args.config}.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        chosen_gpu = min(args.gpu, n_gpu - 1)
        device = f"cuda:{chosen_gpu}"
        print(f"CUDA is available. {n_gpu} GPU(s) detected. Using {device}.")
    else:
        device = "cpu"
        print(f"CUDA is not available. Using {device}.")
    config["DEVICE"] = device
    if not os.path.exists(config["SAVE_PATH"]):
        os.makedirs(config["SAVE_PATH"])
    config["START_TIME"] = start_time()
    print("Start Time:", config["START_TIME"])
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*TypedStorage is deprecated.*"
    )
    seed_everything(config["SEED"])
    wandb.login()
    if "gpt" in config["TRAIN"]["MODEL"].lower():
        config["NAME"] = "kogpt"
    elif "solar" in config["TRAIN"]["MODEL"].lower():
        config["NAME"] = "solar"
    elif "datavortex" in config["TRAIN"]["MODEL"].lower():
        config["NAME"] = "datavortex"
    wandb.init(project="HansolDecoLLM", name=f'{config["NAME"]}_{config["START_TIME"]}')
    main(config)
