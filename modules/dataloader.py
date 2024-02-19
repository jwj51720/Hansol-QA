import pandas as pd
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from modules.utils import *


class CustomDataset(Dataset):
    def __init__(self, data, masks):
        self.data = data
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.masks is not None:
            return self.data[idx], self.masks[idx]
        else:
            return self.data[idx]


class qa_template():
    def __init__(self, CFG):
        train_tokenizer = CFG["TRAIN"]["TOKENIZER"]
        if train_tokenizer == "skt/kogpt2-base-v2":
            template = "/kogpt.txt"
        elif train_tokenizer == "beomi/OPEN-SOLAR-KO-10.7B":
            template = "/bsolar.txt"
        elif train_tokenizer == "LDCC/LDCC-SOLAR-10.7B":
            template = "/ldcc.txt"
        with open(CFG["DATA_PATH"] + template, 'r', encoding='utf-8') as file:
            self.content = file.read()
    def fill(self, q, a):
        if a is None:
            answer_start_index = self.content.find('<answer>')
            content = self.content[:answer_start_index]
        else:
            content = self.content.replace('<question>', q)
            content = content.replace('<answer>', a)
        return content

def train_preprocessing(CFG):
    qa = qa_template(CFG)
    train_tokenizer = CFG["TRAIN"]["TOKENIZER"]
    tokenizer = get_tokenizer(train_tokenizer)
    data = pd.read_csv(f'{CFG["DATA_PATH"]}/{CFG["TRAIN_DATA"]}')
    formatted_data = []
    attention_masks = []
    for _, row in data.iterrows():
        for q_col in ["질문_1", "질문_2"]:
            for a_col in ["답변_1", "답변_2", "답변_3", "답변_4", "답변_5"]:
                input_text = qa.fill(row[q_col], row[a_col]) + tokenizer.eos_token
                encoding = tokenizer(
                    input_text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
                )
                formatted_data.append(encoding["input_ids"].squeeze(0))
                attention_masks.append(encoding["attention_mask"].squeeze(0))

    train_data, valid_data, train_masks, valid_masks = train_test_split(
        formatted_data,
        attention_masks,
        test_size=CFG["TRAIN"]["VALID_SPLIT"],
        random_state=CFG["SEED"],
    )
    save_params(CFG, tokenizer, "tokenizer")
    return train_data, valid_data, train_masks, valid_masks, tokenizer

def get_tokenizer(tokenizer):
    if tokenizer == "skt/kogpt2-base-v2":
        load_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer, eos_token="</s>", pad_token="<pad>"
        )
    elif tokenizer == "beomi/OPEN-SOLAR-KO-10.7B":
        load_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, eos_token="</s>", pad_token="</s>", padding_side="left"
        )
    elif tokenizer == "LDCC/LDCC-SOLAR-10.7B":
        load_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, eos_token="<|im_end|>", pad_token="</s>", padding_side="left"
        )
    return load_tokenizer

def test_preprocessing(CFG):
    qa = qa_template(CFG)
    test_tokenizer = CFG["TRAIN"]["TOKENIZER"]
    tokenizer = get_tokenizer(test_tokenizer)
    data = pd.read_csv(f'{CFG["DATA_PATH"]}/{CFG["TEST_DATA"]}')
    formatted_data = []
    for _, row in data.iterrows():
        input_text = qa.fill(row["질문"], None)
        input_ids = tokenizer.encode(
            input_text, padding=True, return_tensors="pt",add_special_tokens=False
        ).squeeze(0)
        formatted_data.append(input_ids)
    return formatted_data


def get_loader(CFG):
    train_data, valid_data, train_masks, valid_masks, tokenizer = train_preprocessing(
        CFG
    )
    train_dataset = CustomDataset(train_data, train_masks)
    valid_dataset = CustomDataset(valid_data, valid_masks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
        collate_fn=create_collate_fn(tokenizer),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
        collate_fn=create_collate_fn(tokenizer),
    )
    return train_loader, valid_loader


def get_test_loader(CFG):
    test_data = test_preprocessing(CFG)
    test_dataset = CustomDataset(test_data, None)
    test_loader = DataLoader(
        test_dataset, batch_size=CFG["INFERENCE"]["BATCH_SIZE"], shuffle=False
    )
    return test_loader


def create_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks_padded = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        return input_ids_padded, attention_masks_padded

    return collate_fn
