import extract_qa
import pandas as pd
import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import time
import shutil
import re
import json
start_time = time.time()
WORKING_DIR = os.getcwd()
# example: /home/user/graphrag
# use pwd command

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
model_name = "llama3.1:latest"
# start_time = time.time()
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name=model_name,
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
qa = extract_qa.qaExtractor("perguntas_saude.txt", "ground_truth_saude.txt")
dicio = {}
ground_truth = qa.get_second()
perguntas = qa.get_first()
count = 0
for pergunta in perguntas[:95]:
    try:
        print(count)
        resposta = rag.query(f"Responda em Português: {pergunta}", param=QueryParam(mode="local"))
        result = rag.query(f"Responda em Português: {pergunta}", param=QueryParam(mode="local", only_need_context=True))
        if result:
            split_text = result.split('-----Sources-----')
            if len(split_text) > 1:
                csv_content = split_text[1].strip()
                csv_content = csv_content[6:-3]
            with open("context.csv", "w") as file:
                file.write(csv_content)
            df = pd.read_csv('context.csv')
            contexto = df['content'][0]
        else:
            contexto = "no context"
        file_name = "qa.json"
        dicio.update({count : [{"question" : pergunta, "answer" : resposta, "context": contexto, "ground_truth": ground_truth[count]}]})
        count += 1
    except:
        continue
with open(file_name, "w", encoding="utf-8") as json_file:
    json.dump(dicio, json_file, indent=4, ensure_ascii=False)
print(f"JSON data has been saved to {file_name}")