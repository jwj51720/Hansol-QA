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


def train_preprocessing(CFG):
    train_tokenizer = CFG["TRAIN"]["TOKENIZER"]
    if train_tokenizer == "skt/kogpt2-base-v2":
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            train_tokenizer, eos_token="</s>", pad_token="<pad>"
        )
    elif train_tokenizer == "beomi/OPEN-SOLAR-KO-10.7B":
        tokenizer = AutoTokenizer.from_pretrained(
            train_tokenizer, eos_token="</s>", pad_token="</s>"
        )
    data = pd.read_csv(f'{CFG["DATA_PATH"]}/{CFG["TRAIN_DATA"]}')
    formatted_data = []
    attention_masks = []  # 어텐션 마스크를 저장할 리스트
    for _, row in data.iterrows():
        for q_col in ["질문_1", "질문_2"]:
            for a_col in ["답변_1", "답변_2", "답변_3", "답변_4", "답변_5"]:
                input_text = row[q_col] + tokenizer.eos_token + row[a_col]
                # `return_tensors="pt"` 옵션과 함께 `tokenizer` 호출
                encoding = tokenizer(
                    input_text, return_tensors="pt", padding=True, truncation=True
                )
                formatted_data.append(encoding["input_ids"].squeeze(0))  # 배치 차원 제거
                attention_masks.append(
                    encoding["attention_mask"].squeeze(0)
                )  # 배치 차원 제거

    # 학습 데이터와 검증 데이터로 분리
    train_data, valid_data, train_masks, valid_masks = train_test_split(
        formatted_data,
        attention_masks,
        test_size=CFG["TRAIN"]["VALID_SPLIT"],
        random_state=CFG["SEED"],
    )
    save_params(CFG, tokenizer, "tokenizer")
    # 훈련 데이터, 검증 데이터와 함께 어텐션 마스크도 반환
    return train_data, valid_data, train_masks, valid_masks, tokenizer


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
    train_data, valid_data, train_masks, valid_masks, tokenizer = train_preprocessing(
        CFG
    )
    train_dataset = CustomDataset(train_data, train_masks)
    valid_dataset = CustomDataset(valid_data, valid_masks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["TRAIN"]["BATCH_SIZE"],
        shuffle=True,
        collate_fn=create_collate_fn(tokenizer),  # 여기에 collate_fn 지정
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
        collate_fn=create_collate_fn(tokenizer),  # 여기에 collate_fn 지정
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
        # 배치 내의 `input_ids`와 `attention_mask`를 분리
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]

        # `pad_sequence`를 사용하여 각각 패딩 적용
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks_padded = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )  # 패딩된 부분은 0으로 마스킹

        return input_ids_padded, attention_masks_padded

    return collate_fn
