import pandas as pd
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from modules.utils import *


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_preprocessing(CFG):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        CFG["TRAIN"]["TOKENIZER"], eos_token="</s>", pad_token="<pad>"
    )
    data = pd.read_csv(f'{CFG["DATA_PATH"]}/{CFG["TRAIN_DATA"]}')
    formatted_data = []
    for _, row in data.iterrows():
        for q_col in ["질문_1", "질문_2"]:
            for a_col in ["답변_1", "답변_2", "답변_3", "답변_4", "답변_5"]:
                input_text = row[q_col] + tokenizer.eos_token + row[a_col]
                input_ids = tokenizer.encode(input_text, return_tensors="pt").squeeze(
                    0
                )  # Remove the batch dimension
                formatted_data.append(input_ids)
    train_data, valid_data = train_test_split(
        formatted_data, test_size=CFG["TRAIN"]["VALID_SPLIT"], random_state=CFG["SEED"]
    )
    save_params(CFG, tokenizer, "tokenizer")
    return train_data, valid_data, tokenizer


def test_preprocessing(CFG):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(CFG["INFERENCE"]["TOKENIZER"])
    data = pd.read_csv(f'{CFG["DATA_PATH"]}/{CFG["TEST_DATA"]}')
    formatted_data = []
    for _, row in data.iterrows():
        input_text = row["질문"] + tokenizer.eos_token
        input_ids = tokenizer.encode(
            input_text, padding=True, return_tensors="pt"
        ).squeeze(0)
        formatted_data.append(input_ids)
    return formatted_data


def get_loader(CFG):
    train_data, valid_data, tokenizer = train_preprocessing(CFG)
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)

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
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=CFG["INFERENCE"]["BATCH_SIZE"], shuffle=False
    )
    return test_loader


def create_collate_fn(tokenizer):
    def collate_fn(batch):
        batch_padded = pad_sequence(
            batch, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return batch_padded

    return collate_fn