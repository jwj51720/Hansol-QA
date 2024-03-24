from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
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
import GPUtil
from cryptography.fernet import Fernet


def start_time():
    korea_timezone = pytz.timezone("Asia/Seoul")
    current_time_korea = datetime.now(korea_timezone)
    formatted_time = current_time_korea.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_time


def initialize_model(
    model_name, train_model, revision=None, is_training=True, CFG=None
):
    if train_model == "skt/kogpt2-base-v2":
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif train_model in [
        "beomi/OPEN-SOLAR-KO-10.7B",
        "LDCC/LDCC-SOLAR-10.7B",
        "Edentns/DataVortexS-10.7B-dpo-v1.11",
    ]:
        # quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        model = AutoModelForCausalLM.from_pretrained(
            train_model, revision=revision, quantization_config=bnb_config
        )
        if is_training:  # train
            # lora
            lora_config = LoraConfig(
                lora_alpha=CFG["TRAIN"]["LORA"]["ALPHA"],
                lora_dropout=CFG["TRAIN"]["LORA"]["DROPOUT"],
                r=CFG["TRAIN"]["LORA"]["R"],
                bias="none",
                task_type="CAUSAL_LM",
            )

            model.config.use_cache = False
            model.config.pretraining_tp = 1
            model.enable_input_require_grads()
            model = get_peft_model(model, lora_config)
        else:  # inference
            model = PeftModel.from_pretrained(model, model_name)
            model = model.merge_and_unload()
    return model


def get_model(CFG, is_training=True):
    GPUs = GPUtil.getGPUs()
    device = CFG["DEVICE"]
    train_model = CFG["TRAIN"]["MODEL"]

    model_name = (
        CFG["TRAIN"]["MODEL"] if is_training else CFG["INFERENCE"]["TRAINED_MODEL"]
    )

    if is_training:
        if train_model == "beomi/OPEN-SOLAR-KO-10.7B":
            model = initialize_model(
                model_name, train_model=train_model, is_training=is_training, CFG=CFG
            )
        elif train_model == "LDCC/LDCC-SOLAR-10.7B":
            model = initialize_model(
                model_name,
                train_model=train_model,
                revision="v1.1",
                is_training=is_training,
                CFG=CFG,
            )
        elif train_model == "skt/kogpt2-base-v2":
            model = initialize_model(
                model_name, is_training=is_training, train_model=train_model
            )
        elif train_model == "Edentns/DataVortexS-10.7B-dpo-v1.11":
            model = initialize_model(
                model_name, is_training=is_training, train_model=train_model, CFG=CFG
            )
    else:  # inference
        revision = "v1.1" if train_model == "LDCC/LDCC-SOLAR-10.7B" else None
        model = initialize_model(
            model_name,
            is_training=is_training,
            revision=revision,
            train_model=train_model,
        )
    try:
        model = model.to(device)
        for gpu in GPUs:
            print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
            print(f"GPU 사용량: {gpu.load*100}%")
            print(f"GPU 메모리 사용량: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        return model
    except ValueError:
        print(
            "Model is already on device cuda because of bitsandbytes(4bit/8bit) load."
        )
        for gpu in GPUs:
            print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
            print(f"GPU 사용량: {gpu.load*100}%")
            print(f"GPU 메모리 사용량: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        return model


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


def get_name(CFG, type):
    file_name = (
        CFG["TRAIN"]["MODEL"].split("/")[-1]
        if type == "model"
        else CFG["TRAIN"]["TOKENIZER"].split("/")[-1]
    )
    return file_name


def save_params(CFG, params, type="model"):
    file_name = get_name(CFG, type)
    start_time = CFG["START_TIME"]
    params.save_pretrained(f"{CFG['SAVE_PATH']}/{file_name}_{start_time}")


def extract_answer(CFG, outputs):
    start_token = CFG["START_TOKEN"]
    preds = []
    tokenizer = PreTrainedTokenizerFast.from_pretrained(CFG["INFERENCE"]["TOKENIZER"])
    for output_sequences in outputs:
        for generated_sequence in output_sequences:
            full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            answer_start = full_text.find(start_token) + len(start_token)
            answer_only = full_text[answer_start:].strip()
            answer_only = answer_only.replace("\n", " ")
            answer_only = answer_only.replace(tokenizer.pad_token, "")
            answer_only = answer_only.replace(tokenizer.eos_token, "")
            preds.append(answer_only)
    return preds


def submission(CFG, preds):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    nl = pd.read_csv(f"{CFG['DATA_PATH']}/{CFG['TEST_DATA']}")
    submit = pd.read_csv(f"{CFG['DATA_PATH']}/{CFG['SUBMISSION_DATA']}")
    submission_name = CFG["INFERENCE"]["TOKENIZER"].split("/")[-1]
    nl["답변"] = preds
    nl.to_csv(
        f'{CFG["SAVE_PATH"]}/{submission_name}/NL_{submission_name}.csv', index=False
    )
    if len(nl) != len(submit):
        nl = (
            nl.groupby("id")["답변"]
            .apply(lambda x: " ".join(x.astype(str)))
            .reset_index()
        )
        preds = nl["답변"]
        nl.to_csv(
            f'{CFG["SAVE_PATH"]}/{submission_name}/NL_merge_{submission_name}.csv',
            index=False,
        )
    pred_embeddings = model.encode(preds)
    print("Shape of Prediction Embeddings: ", pred_embeddings.shape)
    submit.iloc[:, 1:] = pred_embeddings
    submit.to_csv(
        f'{CFG["SAVE_PATH"]}/{submission_name}/{submission_name}.csv', index=False
    )


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


def model_size_in_mb(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * 4
    total_mb = total_bytes / (1024**2)
    return total_mb
