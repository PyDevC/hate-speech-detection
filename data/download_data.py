from datasets import load_dataset
"""
run:  python3 download_data.py
"""

data = load_dataset("tdavidson/hate_speech_offensive") # change the name to path of dataset from huggingface
data.save_to_disk("tdavidson/hate_speech_offensive")
