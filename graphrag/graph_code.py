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
start_time = time.time()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
# start_time = time.time()
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="gemma2:latest",
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

# Batch insert texts into LightRAG with a retry mechanism
def insert_texts_with_retry(rag, texts, retries=3, delay=2):
    for _ in range(retries):
        try:
            rag.insert(texts)
            return
        except Exception as e:
            print(
                f"Error occurred during insertion: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
    raise RuntimeError("Failed to insert texts after multiple retries.")


while os.listdir("docs"):
    texts = []
    for filename in os.listdir("docs"):
        if filename.endswith(".txt"):
            file_path = os.path.join("docs", filename)
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
            shutil.move(file_path, os.path.join("done", filename))
            break
    if texts:
        insert_texts_with_retry(rag, texts)

# Perform naive search
# print(
#     rag.query("Responda em Português: A partir de quando a gestante deve procurar o serviço de saúde para suplementação de ferro?", param=QueryParam(mode="local"))
# )

# # print(
# #     rag.query("Answer in Portuguese: Who is Vanessa Oliveira and what is her relation with Diego Madureira in the context of Universidade de Brasília?", param=QueryParam(mode="naive", only_need_context=True))
# # )
# # print(
# #     rag.query("Responda em Portuguẽs: Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureira no contexto da Universidade de Brasília?", param=QueryParam(mode="hybrid", only_need_context=True))
# # )
end_time = time.time()
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60
print(f"Execution time: {execution_time_minutes} minutes")
