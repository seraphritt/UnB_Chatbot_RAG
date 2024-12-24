import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = os.getcwd()
# example: /home/user/graphrag
# use pwd command

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="gemma2:2b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
    addon_params={"language": "English"}
)

with open("manual_para_estudantes_2022.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())
# Perform naive search
# print(
#     rag.query("Answer in Portuguese: Who is Vanessa Oliveira and what is her relation with Diego Madureira in the context of Universidade de Brasília?", param=QueryParam(mode="global"))
# )

# print(
#     rag.query("Responda em Portuguẽs: Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureire no contexto da Universidade de Brasília?", param=QueryParam(mode="local"))
# )

# print(
#     rag.query("Answer in Portuguese: Who is Vanessa Oliveira and what is her relation with Diego Madureira in the context of Universidade de Brasília?", param=QueryParam(mode="naive"))
# )
# print(
#     rag.query("Responda em Portuguẽs: Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureire no contexto da Universidade de Brasília?", param=QueryParam(mode="hybrid"))
# )