from dataset.tdavidson import tdavidson_hate_offensive
from datasets import load_dataset, Dataset
from train.base import train_model
from train.model import save_model, load_model
from sklearn.metrics import accuracy_score, r2_score, precision_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

def train(huggingface:str, stratify_column, model_name, device):
    data = load_dataset(huggingface)
    df = data["train"].to_pandas()

    train_df, val_df = train_test_split(df, 
                                        test_size=0.25, 
                                        stratify=df[stratify_column], 
                                        random_state=42
                                        )

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = tdavidson_hate_offensive(train_dataset, tokenizer)
    val_dataset = tdavidson_hate_offensive(val_dataset, tokenizer)


    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0)
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, train_dataset, val_dataset, optimizer, lr_scheduler, criterion)
    save_model(model, model_name)


def test():
    return results, y_test

def results(results, y_test):
    accuracy = accuracy_score(results, y_test)
    r2 = r2_score(results, y_test)
    precision = precision_score(results, y_test)
    print(f"BERT_HateExplain: accuracy: {accuracy}, r2: {r2}, precision: {precision}")

def plot():
    pass

if __name__ == "__main__":
    train()
    results, y_test = test()
    results(results, y_test)
    plot()
