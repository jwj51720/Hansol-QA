from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    train_model = CFG["TRAIN"]["MODEL"]
    lora = CFG["TRAIN"]["LORA"]
    if mode == "inference":
        inference_model = CFG["INFERENCE"]["TRAINED_MODEL"]

    if mode == "train":
        if train_model == "skt/kogpt2-base-v2":
            model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        elif train_model in ["beomi/OPEN-SOLAR-KO-10.7B", "LDCC/LDCC-SOLAR-10.7B"]:
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            # )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                train_model,
                quantization_config=bnb_config,
            )
            lora_config = LoraConfig(
                lora_alpha=lora["ALPHA"],
                lora_dropout=lora["DROPOUT"],
                r=lora["R"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
    elif mode == "inference":
        if train_model == "skt/kogpt2-base-v2":
            model = GPT2LMHeadModel.from_pretrained(inference_model)
        elif train_model in ["beomi/OPEN-SOLAR-KO-10.7B", "LDCC/LDCC-SOLAR-10.7B"]:
            model = AutoModelForCausalLM.from_pretrained(inference_model)
    return model.to(device)


def get_optimizer(CFG, model):
    select_optimizer = CFG["TRAIN"]["OPTIMIZER"]
    learning_rate = CFG["TRAIN"]["LEARNING_RATE"]
    if select_optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer


def get_scheduler(CFG, optimizer):
    select_scheduler = CFG["TRAIN"]["SCHEDULER"]
    select_scheduler_cfg = select_scheduler["CFG"]
    if select_scheduler["NAME"].lower() == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, T_max=select_scheduler_cfg["TMAX"])
    return scheduler


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
            full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            answer_start = full_text.find(tokenizer.eos_token) + len(
                tokenizer.eos_token
            )
            answer_only = full_text[answer_start:].strip()
            answer_only = answer_only.replace("\n", " ")
            answer_only = answer_only.replace("<pad>", "")
            preds.append(answer_only)
    return preds


def submission(CFG, preds):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    nl = pd.read_csv(f"{CFG['DATA_PATH']}/{CFG['TEST_DATA']}")
    submit = pd.read_csv(f"{CFG['DATA_PATH']}/{CFG['SUBMISSION_DATA']}")
    submission_name = CFG["INFERENCE"]["TRAINED_MODEL"].split("/")[-1]
    nl["답변"] = preds
    nl.to_csv(f'{CFG["SAVE_PATH"]}/NL_{submission_name}.csv', index=False)
    if len(nl) != len(submit):
        nl = (
            nl.groupby("id")["답변"]
            .apply(lambda x: " ".join(x.astype(str)))
            .reset_index()
        )
        preds = nl["답변"]
        nl.to_csv(f'{CFG["SAVE_PATH"]}/NL_merge_{submission_name}.csv', index=False)
    pred_embeddings = model.encode(preds)
    print("Shape of Prediction Embeddings: ", pred_embeddings.shape)
    submit.iloc[:, 1:] = pred_embeddings
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
    with open(f"config/{config}", "rb") as file:
        encrypted_config = file.read()
    decrypted_config_bytes = cipher_suite.decrypt(encrypted_config)
    decrypted_config_str = decrypted_config_bytes.decode("utf-8")
    decrypted_config_json = json.loads(decrypted_config_str)
    return decrypted_config_json
