import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import time

start_time = time.time()
WORKING_DIR = os.getcwd()
# example: /home/user/graphrag
# use pwd command

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="mistral:latest",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=lambda texts: ollama_embedding(
        texts, embed_model="bge-m3", host="http://localhost:11434"
        ),
    ),

    addon_params={"language": "Portuguese"}
)

# with open("manual_check_naoob_ob.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())
# Perform naive search
print(
    rag.query("Answer in Portuguese: O que é a FAU?", param=QueryParam(mode="local", only_need_context=True, top_k=3))
)
print(
    rag.query("Answer in Portuguese: What is the FAU?", param=QueryParam(mode="local", top_k=3))
)
# print(
#     rag.query("Responda em Portuguẽs: Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureire no contexto da Universidade de Brasília?", param=QueryParam(mode="local"))
# )

# print(
#     rag.query("Answer in Portuguese: Who is Vanessa Oliveira and what is her relation with Diego Madureira in the context of Universidade de Brasília?", param=QueryParam(mode="naive", only_need_context=True))
# )
# print(
#     rag.query("Responda em Portuguẽs: Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureire no contexto da Universidade de Brasília?", param=QueryParam(mode="hybrid", only_need_context=True))
# )
# end_time = time.time()
# execution_time_seconds = end_time - start_time
# execution_time_minutes = execution_time_seconds / 60
# print(f"Execution time: {execution_time_minutes} minutes")
