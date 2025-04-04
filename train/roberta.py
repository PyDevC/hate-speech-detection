import torch
import random
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def train(dataset_name, text, label, model_name):
    dataset = load_dataset(dataset_name)

    df = dataset["train"].to_pandas()

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[label], random_state=42)

    # Convert back to Hugging Face Dataset (drop index to avoid issues)
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    class HateSpeechDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=128):
            self.texts = list(data[text]) 
            self.labels = list(data[label])  
            self.tokenizer = tokenizer
            self.max_length = max_length
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            if isinstance(idx, (list, tuple)):  
                texts = [self.texts[i] for i in idx]
                labels = [self.labels[i] for i in idx]
            else:
                texts = [self.texts[idx]]
                labels = [self.labels[idx]]
    
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
    
            return {
                "input_ids": encodings["input_ids"].squeeze(0),  
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
    
    
    train_data = HateSpeechDataset(train_dataset, tokenizer)
    val_data = HateSpeechDataset(val_dataset, tokenizer)  
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    
    # Load RoBERTa model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    criterion = nn.CrossEntropyLoss()
    
    def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=3):
        model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            loop = tqdm(train_loader, leave=True)
    
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
    
            acc, f1, r2 = evaluate_model(model, val_loader)
        return acc, f1, r2
    
    def evaluate_model(model, val_loader):
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
        print(f"Validation Loss: {total_loss / len(val_loader):.4f}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        return acc, f1, r2
    
    
    acc, f1, r2= train_model(model, train_loader, val_loader, optimizer, criterion, epochs=3)
    return acc, f1, r2
    
def run():
    data = { "model": [],
            "dataset": [],
            "accuracy": [],
            "f1 score": [], 
            "r2_score": []
    }
    datasets = [
            "tdavidson/hate_speech_offensive",
            "limjiayi/hateful_memes_expanded",
        "community-datasets/roman_urdu_hate_speech"
    ]

    models = ["GroNLP/hateBERT", "unitary/toxic-bert", "google-bert/bert-base-uncased", "FacebookAI/roberta-large"]

    random.shuffle(models)
    random.shuffle(datasets)
    for i in datasets:
        for j in models:
            print(i, j)
            text, label = input().split()
            data["model"].append(j)
            data["dataset"].append(i)
            acc, f1, r2 = train(i, text, label ,j)
            data["accuracy"].append(acc)
            data["f1 score"].append(f1)
            data["r2_score"].append(r2)
            print(data)
    df = pd.DataFrame(data)
    df.to_csv("results.csv")
run()
