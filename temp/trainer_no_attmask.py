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
        for batch_idx, batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        valid_loss = validation(CFG, model, valid_loader)
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


def validation(CFG, model, valid_loader):
    device = CFG["DEVICE"]
    total_loss = 0
    model.eval()
    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
    valid_loss = total_loss / len(valid_loader)
    return valid_loss
