import torch
from sklearn.metrics import accuracy_score, r2_score, precision_score, f1_score
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, train_dataset, val_dataset, optimizer, criterion):
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
    num_training_steps = len(train_dataloader) 
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    epochs = 3
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_dataloader, leave=True)

        for batch in loop:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if total > 0: 
                loop.set_postfix(loss=total_loss / total, acc=correct / total)
        print(f"epoch[{epoch}]/[{epochs}] completed")

        evaluate_model(model, val_loader, criterion,device)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0
    print("evaluation")
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
    print(f"Validation Loss: {total_loss / len(val_loader):.4f}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}, r2_score: {r2:.4f}")
