from dataset import tdavidson, hatexplain 
import torch
from datasets import load_dataset, Dataset
from train.base import train_model
from sklearn.metrics import accuracy_score, r2_score, precision_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

def train(huggingface:str, stratify_column, model_name, device):
    data = load_dataset(huggingface)
    df = data["test"].to_pandas()

    train_df, val_df = train_test_split(df, 
                                        test_size=0.25, 
                                        stratify=df[stratify_column], 
                                        random_state=42
                                        )

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = tdavidson.tdavidson_hate_offensive(train_dataset, tokenizer)
    val_dataset = tdavidson.tdavidson_hate_offensive(val_dataset, tokenizer)


    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, train_dataset, val_dataset, optimizer, criterion)
    model.save_pretrained("models/bert_tdavidson")

if __name__ == "__main__":

    huggingface = "tweets-hate-speech-detection/tweets_hate_speech_detection"
    stratify_column = "label"
    model_name = "FacebookAI/roberta-base"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(huggingface, stratify_column, model_name, device)
