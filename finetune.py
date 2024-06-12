from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from project_variables import SYTHETIC_DATA_DIR
import os
import json

with open(os.path.join(SYTHETIC_DATA_DIR, 'train_dataset.json')) as json_data:
    dataset = json.load(json_data)
    train_dataset = EmbeddingQAFinetuneDataset(queries=dataset['queries'], 
                                               corpus=dataset['corpus'],
                                               relevant_docs=dataset['relevant_docs'])
    json_data.close()
with open(os.path.join(SYTHETIC_DATA_DIR, 'val_dataset.json')) as json_data:
    dataset = json.load(json_data)
    val_dataset = EmbeddingQAFinetuneDataset(queries=dataset['queries'], 
                                             corpus=dataset['corpus'],
                                             relevant_docs=dataset['relevant_docs'])
    json_data.close()

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-m3",
    model_output_path="finetuned_model",
    val_dataset=val_dataset,
)

finetune_engine.finetune()