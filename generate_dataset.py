from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

import os
import json
import random
import json
from deep_translator import GoogleTranslator

from project_variables import DATA_DIR, SYTHETIC_DATA_DIR, MODEL_NAME, HF_TOKEN
from prompt_templates import QA_GENERATE_PROMPT_TMPL
from models import get_llm

LLM = get_llm(MODEL_NAME, HF_TOKEN)


class MyJSONReader(BaseReader):
    def split_path(self, path):
        # Split the path into directory and filename components
        directory, filename = os.path.split(path)
        # Split the directory into its parent directory and folder name
        parent_directory, folder_name = os.path.split(directory)
        return parent_directory, folder_name, filename

    def load_data(self, file, extra_info=None):
        parent_directory, folder_name, filename = self.split_path(str(file))
        documents = []
        with open(file, "r") as f:
            docs = json.load(f)
        for doc in docs:
            text = ''
            metadata = {}
            if filename=='english.json':
                text = doc['Text']
                metadata = {
                    'article_id': doc['ArticleId'],
                    'category': doc['Category'],
                }
            else:
                text = doc['content']
                metadata = {
                    'article_id': doc['seq'],
                    'title': doc['title'],
                    'date': doc['date'],
                    'article_url': doc['article_url'],
                    'label1': doc['label1'],
                    'label2': doc['label2'],
                }
            documents.append(Document(text=text, metadata=metadata))
        return documents

def node_parser(docs):
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs)
    return nodes

def read_data(data_file):
    reader = SimpleDirectoryReader(input_files=[data_file], file_extractor={".json": MyJSONReader()})
    documents = reader.load_data()
    nodes = node_parser(documents)
    return nodes

def generate_training_data(data_dir):
    en_nodes = read_data(os.path.join(data_dir, 'english.json'))
    ko_nodes = read_data(os.path.join(data_dir, 'korean.json'))
    train_nodes = en_nodes[:1000] + ko_nodes[:6000]
    val_nodes = en_nodes[1000:] + ko_nodes[6000:]
    return train_nodes, val_nodes

def filter_queries_by_relevant_docs(doc_id, relevant_docs):
    filtered_dict = dict(
        filter(lambda item: item[1][0]==doc_id, relevant_docs.items())
    )
    return list(filtered_dict.keys())

def translate_data(dataset):
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    doc_ids = list(dataset.corpus.keys())

    for doc_id in doc_ids:
        query_ids = filter_queries_by_relevant_docs(doc_id, relevant_docs)
        random_ids = random.sample(query_ids, int(len(query_ids)/2))
        for id in random_ids:
            translated_query = GoogleTranslator(target='ko').translate(queries[id])
            queries[id] = translated_query
    for id, query in queries.items():
        if query is None:
            queries[id] = "None"
    return EmbeddingQAFinetuneDataset(queries=queries, corpus=dataset.corpus, relevant_docs=relevant_docs)

def generate_synthetic_data(data_dir, save_path, llm, prompt_tmpl,):
    train_nodes, val_nodes = generate_training_data(data_dir)
    train_dataset = generate_qa_embedding_pairs(
        llm=llm,
        num_questions_per_chunk=4,
        nodes=train_nodes,
    )
    train_dataset = translate_data(train_dataset)
    val_dataset = generate_qa_embedding_pairs(
        llm=llm,
        num_questions_per_chunk=4,
        nodes=val_nodes,
    )
    val_dataset = translate_data(val_dataset)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    train_dataset.save_json(os.path.join(save_path, "train_dataset.json"))
    val_dataset.save_json(os.path.join(save_path, "val_dataset.json"))

generate_synthetic_data(DATA_DIR, SYTHETIC_DATA_DIR, LLM, QA_GENERATE_PROMPT_TMPL)