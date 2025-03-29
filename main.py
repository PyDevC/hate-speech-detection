from dataset.hate_explain import HateExplain
from train import BERT, HateBert, Roberta, HateCheck
from torch.optim import Adam
from train.base import train_model
from train.model import save_model, load_model
from sklearn.metrics import accuracy_score, r2_score, precision_score

def train():
    model = BERT.bert()
    dataset = HateExplain("")
    optimizer = Adam(model.parameters(), lr=0.0001)
    model = train_model(model, dataset, optimizer)
    save_model(model, "BERT_HateExplain.pt")

def test():
    model = BERT.bert()
    model = load_model(model, "BERT_HateExplain.pt")
    X_test = HateExplain("").X_test
    y_test = HateExplain("").y_test
    results = model.predict(X_test)
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
