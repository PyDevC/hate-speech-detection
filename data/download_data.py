from datasets import load_dataset
import sys
"""
run: python3 download_data.py tdavidson/hate_speech_offensive
make sure the file downloads only in parquet form
fix all issues
"""

data = load_dataset(sys.argv[0])
data.to_parquet("data.parquet")
