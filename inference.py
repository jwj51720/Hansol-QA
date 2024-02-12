import os
import torch
import json
import argparse
from tqdm import tqdm
from modules import *


def main(CFG):
    print("**START**")
    model = get_model(CFG, "inference")
    test_loader = get_loader(CFG)
    print("**LOAD DATA COMPLETE**")
    inference(CFG, model, test_loader)
    print("**MODEL INFERENCE COMPLETE**")
    return 0


def inference(CFG, model, test_loader):
    config = CFG["INFERENCE"]
    outputs = []
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            output_sequences = model.generate(
                input_ids=batch.to(device),
                max_length=config["MAX_LENGTH"],
                temperature=config["TEMPERATURE"],
                top_k=config["TOP_K"],
                top_p=config["TOP_P"],
                repetition_penalty=config["REPETITION_PENALTY"],
                do_sample=config["DO_SAMPLE"],
                num_return_sequences=config["NUM_RETURN_SEQUENCES"],
            )
            outputs.append(output_sequences)
        preds = extract_answer(CFG, outputs)
    submission(CFG, preds)


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
    if not os.path.exists(config["SAVE_PATH"]):
        os.makedirs(config["SAVE_PATH"])
    main(config)
