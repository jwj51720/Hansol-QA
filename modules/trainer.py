import torch
from tqdm import tqdm
from modules.utils import *
import wandb


def training(CFG, model, train_loader, valid_loader):
    device = CFG["DEVICE"]
    epochs = CFG["TRAIN"]["EPOCHS"]
    es_patient = CFG["TRAIN"]["EARLY_STOPPING"]
    es_count = 1
    best_loss = float("inf")
    gradient_accumulation_steps = CFG["TRAIN"]["ACCUMUL_STEPS"]
    optimizer = get_optimizer(CFG, model)
    scheduler = get_scheduler(CFG, optimizer)
    for epoch in range(epochs):
        print(f"..Epoch {epoch+1}/{epochs}..")
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (input_ids, attention_mask) in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
        valid_loss = validation(model, valid_loader, device)
        scheduler.step()

        if valid_loss < best_loss:
            es_count = 1
            best_loss = valid_loss
            print("Best Loss Updated. New Best Model Saved.")
            save_params(CFG, model, "model")
        else:
            print(f"Eearly Stopping Count: {es_count}/{es_patient}")
            es_count += 1
        if es_count >= es_patient:
            print(
                "Early stopping patient {es_patient} has been reached, validation loss has not been updated, ending training."
            )
            return 1
        print(f"Train Loss: {total_loss / len(train_loader)}, Valid Loss: {valid_loss}")
        wandb.log(
            {"Train loss": total_loss / len(train_loader), "Valid Loss": valid_loss},
            step=epoch,
        )
    return 0



def validation(model, valid_loader, device):
    total_loss = 0
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for batch_idx, (input_ids, attention_mask) in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss

            total_loss += loss.item()

    valid_loss = total_loss / len(valid_loader)
    return valid_loss
