import torch
from tqdm import tqdm
from modules.utils import *


def training(CFG, model, train_loader, valid_loader):
    device = CFG["DEVICE"]
    epochs = CFG["TRAIN"]["EPOCHS"]
    es_patient = CFG["TRAIN"]["EARLY_STOPPING"]
    es_count = 1
    total_loss = 0
    best_loss = float("inf")

    optimizer = get_optimizer(CFG, model)
    for epoch in range(epochs):
        print(f"..Epoch {epoch+1}/{epochs}..")
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (input_ids, attention_mask) in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )  # labels 수정
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        valid_loss = validation(model, valid_loader, device)

        if valid_loss < best_loss:
            es_count = 1
            best_loss = valid_loss
            save_params(CFG, model, "model")
        else:
            es_count += 1

        if es_count == es_patient:
            print(
                "Early stopping patient {es_patient} has been reached, validation loss has not been updated, ending training."
            )
            return 1
        print(f"Train Loss: {total_loss / len(train_loader)}, Valid Loss: {valid_loss}")
    return 0


from tqdm import tqdm
import torch


def validation(model, valid_loader, device):
    total_loss = 0
    model.eval()  # 모델을 평가 모드로 설정

    with torch.no_grad():  # 그래디언트 계산 비활성화
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for batch_idx, (input_ids, attention_mask) in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 모델 실행 및 손실 계산
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss

            total_loss += loss.item()

    valid_loss = total_loss / len(valid_loader)  # 평균 손실 계산
    return valid_loss
