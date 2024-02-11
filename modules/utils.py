from transformers import GPT2LMHeadModel, AdamW


def get_model(CFG):
    device = CFG["DEVICE"]
    select_model = CFG["MODEL"]
    if select_model == "skt/kogpt2-base-v2":
        model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    return model.to(device)


def get_optimizer(CFG, model):
    select_optimizer = CFG["OPTIMIZER"]
    learning_rate = CFG["LEARNING_RATE"]
    if select_optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer


def evaluation():
    pass
