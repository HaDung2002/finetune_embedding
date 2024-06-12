import os
from models import get_llm

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJECT_PATH, 'data')
SYTHETIC_DATA_DIR = os.path.join(PROJECT_PATH, 'synthetic_data')
HF_TOKEN = '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'
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"