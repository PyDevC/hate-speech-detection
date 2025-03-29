import torch
from torch.utils.data import DataLoader


def train_model(model, train_dataset, optimzer):
    """trains model over several num_epoch
    parameters:
        model: base model or algorithm
        train_dataset: training dataset from dataset.class
        tokenizer: tokenizer

    returns:
        model: trained model
    """
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    num_epoch = 25
    for epoch in range(num_epoch):
        model.train()
        for batch in train_dataloader:
            optimzer.zero_grad()

            input = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()

            optimzer.step()
        
        print(f"Epoch {epoch+1}/{num_epoch} completed!")
    return model
