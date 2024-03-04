import os
import torch
import json
import warnings
import argparse
from tqdm import tqdm
from modules import *


def main(CFG):
    print("**START**")
    model = get_model(CFG, is_training=False)
    test_loader = get_test_loader(CFG)
    print("**LOAD DATA COMPLETE**")
    inference(CFG, model, test_loader)
    print("**MAKE SUBMISSION CSV COMPLETE**")
    return 0


def inference(CFG, model, test_loader):
    device = CFG["DEVICE"]
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
        print("**MODEL INFERENCE COMPLETE**")
        preds = extract_answer(CFG, outputs)
    print("**ANSWER EXTRACT COMPLETE**")
    submission(CFG, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Team 3 - Hansoldeco")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config File Path",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=2,
        help="GPU number you want to use",
    )
    parser.add_argument(
        "-t",
        "--trained",
        type=str,
        required=True,
        help="Trained Model Name you want to inference",
    )
    parser.add_argument(
        "--start_token",
        type=str,
        required=True,
        help="Inference start token",
    )
    args = parser.parse_args()
    # config = crypto_decode(args.config)
    with open(f'config/{args.config}.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    config["INFERENCE"].update(
        {
            "TRAINED_MODEL": f"result/{args.trained}/best",
            "TOKENIZER": f"result/{args.trained}",
        }
    )
    config["START_TOKEN"] = args.start_token
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

    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*TypedStorage is deprecated.*"
    )
    seed_everything(config["SEED"])
    config["PAD_LOC"] = "BACK" if "gpt" in config["TRAIN"]["MODEL"].lower() else "HEAD"
    main(config)
