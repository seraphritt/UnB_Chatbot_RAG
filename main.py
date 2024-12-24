import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain import PromptTemplate
import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    context_entity_recall,

)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import extract_qa
import re
from langchain.document_loaders import PyMuPDFLoader

def load_pdf_data(file_paths):
    all_docs = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def split_docs(documents, chunk_size=700, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda', 'trust_remote_code': True},
        encode_kwargs={
            'normalize_embeddings': normalize_embedding
        }
    )

def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore


template = """
### System:
You are a respectful and honest assistant specialized to answer ONLY about University of Brasília, don't use greetings or saudations. Elaborate your answer with details. \
All your answers from now on must be in Portuguese. \
If the question is not related to the University field, don't answer \
If the answer is not given in the context, say: "Desculpe, mas eu não sei te responder".
Given the following context, answer the following User Question: \

### Context:
{context}

### User Question:
{question}

### Response:
"""

def get_response(retriever, query, template, llm):
    context = retriever.invoke(query)[0].page_content
    # print("RESPOSTA")
    return [llm.invoke(template.format(context=context, question=query)), context]

llm = Ollama(model="qwen2.5:latest", temperature=0.2)
embed = load_embedding_model(model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# List of PDF files to be processed
pdf_files = ["docs/manual_dos_estudantes_22.pdf", "docs/check_list_calouro.pdf", "docs/manual_estagio_curricular_obrigatorio_discentes.pdf", "docs/manual_estagio_nao_obrigatorio_discentes.pdf"]
# Sou calouro, preciso fazer matrícula?
# Loading and splitting the documents from multiple PDF files
docs = load_pdf_data(file_paths=pdf_files)
documents = split_docs(documents=docs)
# Creating vectorstore
vectorstore = create_embeddings(documents, embed)

qa = extract_qa.qaExtractor("ground_truth.txt", "perguntas.txt")
questions = qa.get_questions()
ground_truth = qa.get_answers()
# Converting vectorstore to a retriever
# search_type= similarity (uses l2 (Euclidian Distance) as default)) search_kwargs = k: 3 (take the top 3 results of the similarity search)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Creating the prompt from the template
prompt = PromptTemplate.from_template(template)
count = 0
dicio = {}
# print(get_response(retriever, "Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureire no contexto da Universidade de Brasília?", template, llm))
print("Respondendo questões...")
for entrada in questions:
    answer, contexto = get_response(retriever, entrada, template, llm)
    dicio.update({count : [{"question" : entrada, "answer" : answer, "context": contexto, "ground_truth": ground_truth[count]}]})
    count += 1
    # results = vectorstore.similarity_search(query, k=3)
    # print(f"Retrieved {len(results)} results for the query:")
    # for i, result in enumerate(results):
    #     print(f"Result {i+1}:")
    #     print(f"Content: {result.page_content}")
file_name = "qa.json"
with open(file_name, "w", encoding="utf-8") as json_file:
    json.dump(dicio, json_file, indent=4, ensure_ascii=False)
print(f"JSON data has been saved to {file_name}")
# read json file
with open(file_name, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)