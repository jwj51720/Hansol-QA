from transformers import GPT2LMHeadModel, AdamW
from datetime import datetime
import pytz

def start_time():
    korea_timezone = pytz.timezone('Asia/Seoul')
    current_time_korea = datetime.now(korea_timezone)
    formatted_time = current_time_korea.strftime('%Y:%m:%d-%H:%M:%S')
    return formatted_time

def get_model(CFG, mode="train"):
    device = CFG["DEVICE"]
    if mode == "train":
        select_model = CFG["MODEL"]
        if select_model == "skt/kogpt2-base-v2":
            model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    elif mode == "inference":
        select_model = CFG["INFERENCE"]["TRAINED_MODEL"]
    return model.to(device)


def get_optimizer(CFG, model):
    select_optimizer = CFG["OPTIMIZER"]
    learning_rate = CFG["LEARNING_RATE"]
    if select_optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer

def save_model(CFG, params, type="model"):
    file_name = CFG['MODEL'].split("/")[-1] if type == "model" else CFG['TOKENIZER'].split("/")[-1]
    start_time = CFG['START_TIME']
    params.save_pretrained(f"{CFG['SAVE_PATH']}/{file_name}_{start_time}")

