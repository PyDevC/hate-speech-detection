import torch
from sklearn.metrics import accuracy_score, r2_score, precision_score, f1_score
from torch.utils.data import DataLoader


def train_model(model, train_dataset, val_dataset, optimzer, lr_scheduler, criterion):
    """trains model over several num_epoch
    parameters:
        model: base model or algorithm
        train_dataset: training dataset from dataset.class
        tokenizer: tokenizer

    returns:
        model: trained model
    """
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    num_epoch = 3
    for epoch in range(num_epoch):
        model.train()
        for batch in train_dataloader:
            optimzer.zero_grad()

            input = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input, attention_mask=attention_mask, labels=labels)

            loss = outputs.logits
            loss.backward()
            optimzer.step()
        
            lr_scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epoch} completed!")

    evaluate_model(model, val_loader, criterion, device)
    return model

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="weighted")
    r2 = r2_score(true_labels, preds)
    print(f"Validation Loss: {total_loss / len(val_loader):.4f}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}, r2_score: {r2}")
