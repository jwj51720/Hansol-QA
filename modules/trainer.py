import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from modules import get_optimizer


def trainer(CFG, model, train_loader, valid_loader):
    device = CFG["DEVICE"]
    epochs = CFG["EPOCHS"]
    learning_rate = CFG["LEARNING_RATE"]
    optimizer = get_optimizer(model, learning_rate)

    for epoch in range(epochs):
        print(f"..Epoch {epoch+1}/{epochs}..")
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        for batch_idx, batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        valid_total_loss = validation(CFG, model, valid_loader)
    print(
        f"Train Loss: {total_loss / len(train_loader)}, Valid Loss: {valid_total_loss / len(valid_loader)}"
    )


def validation(CFG, model, valid_loader):
    device = CFG["DEVICE"]
    total_loss = 0
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산을 비활성화
        for batch in valid_loader:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss
