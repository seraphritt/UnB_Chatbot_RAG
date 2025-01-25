import extract_qa
import pandas as pd

df = pd.read_csv('resultRagasEval.csv')
# print(df.keys)
with open('ground_truth.txt', 'w') as file:
    for each in df['reference']:
        file.write(f'{each[1:-1]}\n\n')
# qa = extract_qa.qaExtractor("ground_truth.txt", "perguntas.txt")
# questions = qa.get_questions()
# ground_truth = qa.get_answers()

import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import time
import shutil

start_time = time.time()
WORKING_DIR = os.getcwd()
# example: /home/user/graphrag
# use pwd command

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
# start_time = time.time()
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:latest",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=lambda texts: ollama_embedding(
        texts, embed_model="jeffh/intfloat-multilingual-e5-large-instruct:f16", host="http://localhost:11434"
        ),
    ),

    addon_params={"language": "English"}
)

start_time = time.time()
print(
    rag.query("Responda em Português: A partir de quando a gestante deve procurar o serviço de saúde para suplementação de ferro?", param=QueryParam(mode="local"))
)
print(
    rag.query("Responda em Português: Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureira no contexto da Universidade de Brasília?", param=QueryParam(mode="local", only_need_context=True))
)
end_time = time.time()
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
print(f"Execution time: {execution_time_minutes} minutes")