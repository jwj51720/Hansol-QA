import pandas as pd
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from modules.utils import *


class CustomDataset(Dataset):
    def __init__(self, data, masks, is_hftrainer=True):
        self.data = data
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.masks is not None:
            item = {
                "input_ids": self.data[idx],
                "attention_mask": self.masks[idx],
                "labels": self.data[idx],
            }
            return item
        else:
            item = self.data[idx]
            return item


class QATemplate:
    def __init__(self, CFG):
        train_tokenizer = CFG["TRAIN"]["TOKENIZER"]
        if train_tokenizer == "skt/kogpt2-base-v2":
            template = "/kogpt.txt"
        elif train_tokenizer == "beomi/OPEN-SOLAR-KO-10.7B":
            template = "/bsolar.txt"
        elif train_tokenizer == "LDCC/LDCC-SOLAR-10.7B":
            template = "/ldcc.txt"
        elif train_tokenizer == "Edentns/DataVortexS-10.7B-dpo-v1.11":
            template = "/datavortex.txt"
        with open("template/" + template, "r", encoding="utf-8") as file:
            self.content = file.read()
        self.category_info = {
            "건축구조": "여러 가지 건축 재료를 이용하여 건축물을 형성하는 일 또는 그 건축물에 관한 질문입니다.",
            "마감재": "건물의 겉면을 마감하는 데 쓰는 재료 및 외부의 여러 가지 영향으로부터 건물을 보호하는 것에 관한 질문입니다.",
            "마감하자": "건물의 겉면을 마감하는 데 쓰는 재료 및 건물 보호 재료에 생기는 문제에 관한 질문입니다.",
            "시공": "공사를 시행하면서 사용하는 재료나 방법에 관한 질문입니다.",
            "인테리어": "실내를 장식하는 일이나 실내 장식용품에 관한 질문입니다.",
            "타 마감하자": "표면에 물방울이 맺혀 문제가 생기는 결로 등 생활하면서 생기는 문제에 관한 질문입니다.",
            "기타" : "집 내부와 생활 기준 및 건축의 포괄적인 분야에 관한 질문입니다."
        }

    def fill(self, q, a, c):
        if a is None:
            answer_start_index = self.content.find("<answer>")
            content = self.content[:answer_start_index]
            content = content.replace("<question>", q.strip().replace('"',""))
        else:
            content = self.content.replace("<question>", q.strip().replace('"',""))
            content = content.replace("<answer>", a.strip().replace('"',""))
        # content = content.replace("<category>", self.category_info.get(c))
        return content


def train_preprocessing(CFG):
    qa = QATemplate(CFG)
    train_tokenizer = CFG["TRAIN"]["TOKENIZER"]
    tokenizer = get_tokenizer(train_tokenizer, is_train=True)
    data = pd.read_csv(f'{CFG["DATA_PATH"]}/{CFG["TRAIN_DATA"]}')
    columns_with_question = [col for col in data.columns if "질문" in col]
    columns_with_answer = [col for col in data.columns if "답변" in col]
    formatted_data = []
    attention_masks = []
    for _, row in data.iterrows():
        for q_col in columns_with_question:
            for a_col in columns_with_answer:
                input_text = qa.fill(row[q_col], row[a_col], row['category']) + tokenizer.eos_token
                encoding = tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=CFG["TRAIN"]["MAX_SEQ_LEN"],
                    truncation=True,
                    add_special_tokens=False,
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


def test_preprocessing(CFG):
    qa = QATemplate(CFG)
    test_tokenizer = CFG["INFERENCE"]["TOKENIZER"]
    tokenizer = get_tokenizer(test_tokenizer, is_train=False)
    data = pd.read_csv(f'{CFG["DATA_PATH"]}/{CFG["TEST_DATA"]}')
    formatted_data = []
    for _, row in data.iterrows():
        input_text = qa.fill(row["질문"], None, None)
        input_ids = tokenizer.encode(
            input_text, padding=False, return_tensors="pt", add_special_tokens=False
        ).squeeze(0)
        formatted_data.append(input_ids)
    return formatted_data


def get_tokenizer(tokenizer, is_train):
    if is_train:
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
                tokenizer,
                revision="v1.1",
                eos_token="<|im_end|>",
                pad_token="</s>",
                padding_side="right",
            )
        elif tokenizer == "Edentns/DataVortexS-10.7B-dpo-v1.11":
            load_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, eos_token="<|im_end|>", pad_token="</s>", padding_side="right"
            )
    else:
        load_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return load_tokenizer


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
    return train_dataset, valid_dataset, train_loader, valid_loader


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
