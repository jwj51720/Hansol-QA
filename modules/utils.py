from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from datetime import datetime
import pandas as pd
import pytz
import random
import torch
import numpy as np
import os
import json
from cryptography.fernet import Fernet


def start_time():
    korea_timezone = pytz.timezone("Asia/Seoul")
    current_time_korea = datetime.now(korea_timezone)
    formatted_time = current_time_korea.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_time


def get_model(CFG, mode="train"):
    device = CFG["DEVICE"]
    if mode == "train":
        select_model = CFG["TRAIN"]["MODEL"]
        if select_model == "skt/kogpt2-base-v2":
            model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    elif mode == "inference":
        select_model = CFG["INFERENCE"]["TRAINED_MODEL"]
        model = GPT2LMHeadModel.from_pretrained(select_model)
    return model.to(device)


def get_optimizer(CFG, model):
    select_optimizer = CFG["TRAIN"]["OPTIMIZER"]
    learning_rate = CFG["TRAIN"]["LEARNING_RATE"]
    if select_optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer


def save_params(CFG, params, type="model"):
    file_name = (
        CFG["TRAIN"]["MODEL"].split("/")[-1]
        if type == "model"
        else CFG["TRAIN"]["TOKENIZER"].split("/")[-1]
    )
    start_time = CFG["START_TIME"]
    params.save_pretrained(f"{CFG['SAVE_PATH']}/{file_name}_{start_time}")


def extract_answer(CFG, outputs):
    preds = []
    tokenizer = PreTrainedTokenizerFast.from_pretrained(CFG["INFERENCE"]["TOKENIZER"])
    for output_sequences in outputs:
        for generated_sequence in output_sequences:
            full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
            answer_start = full_text.find(tokenizer.eos_token) + len(
                tokenizer.eos_token
            )
            answer_only = full_text[answer_start:].strip()
            answer_only = answer_only.replace("\n", " ")
            preds.append(answer_only)
    return preds


def submission(CFG, preds):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    pred_embeddings = model.encode(preds)
    print("Shape of Prediction Embeddings: ", pred_embeddings.shape)
    submit = pd.read_csv(f"{CFG['DATA_PATH']}/{CFG['SUBMISSION_DATA']}")
    submit.iloc[:, 1:] = pred_embeddings
    submission_name = CFG["INFERENCE"]["TRAINED_MODEL"].split("/")[-1]
    submit.to_csv(f'{CFG["SAVE_PATH"]}/{submission_name}.csv', index=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def crypto_decode(config):
    key = input("Input Key String: ").encode()
    cipher_suite = Fernet(key)
    with open(config, "rb") as file:
        encrypted_config = file.read()
    decrypted_config_bytes = cipher_suite.decrypt(encrypted_config)
    decrypted_config_str = decrypted_config_bytes.decode("utf-8")
    decrypted_config_json = json.loads(decrypted_config_str)
    return decrypted_config_json
