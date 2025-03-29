import torch
import os

def save_model(model, name):
    torch.save(model.state_dict(), os.path.join("..", "models", name)) 

def load_model(model, name):
    model.load_state_dict(torch.load(os.path.join("..", "models", name), weights_only=True))
    model.eval()
    return model
