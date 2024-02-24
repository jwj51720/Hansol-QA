import torch
from tqdm import tqdm
from modules.utils import *
import wandb
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer


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


class HFTraining:
    def __init__(self, CFG, lora_config, tokenizer) -> None:
        self.CFG = CFG
        self.train_cfg = CFG["TRAIN"]
        self.training_args = TrainingArguments(
            output_dir=CFG["SAVE_PATH"],
            per_device_train_batch_size=16,
            per_device_eval_batch_size = 16,
            gradient_accumulation_steps=1,
            learning_rate = self.train_cfg["LEARNING_RATE"],
            optim="adamw_torch",
            fp16=False,
            bf16=False,
            gradient_checkpointing=True,
            save_strategy="epoch",
            logging_dir="./logs",
            evaluation_strategy="epoch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            load_best_model_at_end=True,
            report_to=["wandb"],
            run_name=f"{self.CFG['NAME']}_{self.CFG['START_TIME']}",
            group_by_length=True
        )
        self.lora_config = lora_config
        self.tokenizer = tokenizer

    def run(self, model, train_dataset, eval_dataset):
        breakpoint()
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.train_cfg["EARLY_STOPPING"])],
        )

        # trainer = SFTTrainer(
        #     model=model,
        #     train_dataset=train_dataset,
        #     peft_config=self.lora_config,
        #     dataset_text_field="QnA",
        #     max_seq_length=1024,
        #     tokenizer=self.tokenizer,
        #     args=self.training_args,
        #     packing=False,
        # )
        trainer.train()
        trainer.save_model()
        return trainer