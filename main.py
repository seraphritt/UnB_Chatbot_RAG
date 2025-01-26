import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain import PromptTemplate
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import FastEmbedEmbeddings
import matplotlib.pyplot as plt
import extract_qa
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
    soma = 0
    for chunk in chunks:
        soma += len(chunk.page_content)
    print("Total de tokens armazenados")
    print(soma) 
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
    print("Tamanho da Vector Store")
    print(vectorstore.index.ntotal)
    return vectorstore


template = """
### System:
You are a respectful and honest assistant specialized to answer ONLY about University of Brasília, don't use greetings or saudations. Elaborate your answer with details. \
All your answers from now on must be in Portuguese. \
If the question is not related to the University field, you cannot answer. \
Only use the given context to develop your answer. \
Given the following context, answer the following User Question: \

### Context:
{context}

### User Question:
{question}

### Response:
"""

def get_response(retriever, query, template, llm):
    retrieved = retriever.invoke(query)
    context = ""
    for each in retrieved:
        context += each.page_content
    # print("RESPOSTA")
    return [llm.invoke(template.format(context=context, question=query)), context]

llm = Ollama(model="llama3.1:latest", temperature=0.1)
embed = LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large'))
# List of PDF files to be processed
pdf_files = [
    "docs/caderno_34" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_13_canceres_do_colo_do_utero_e_da_mamapdf"  + ".pdf",
    "docs/cadernos_de_atencao_basica_no_19_envelhecimento_e_saude_da_pessoa_idosapdf"  + ".pdf",
    "docs/cadernos_de_atencao_basica_no_20_carencias_de_micronutrientespdf"  + ".pdf",
    "docs/cadernos_de_atencao_basica_no_23_saude_da_crianca_aleitamento_materno_e_alimentacao_complementarpdf"  + ".pdf",
    "docs/cadernos_de_atencao_basica_no_29_rastreamentopdf" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_32_atencao_ao_pre_natal_de_baixo_riscopdf" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_33_saude_da_crianca_crescimento_e_desenvolvimentopdf" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_35_estrategias_para_o_cuidado_da_pessoa_com_doenca_cronicapdf" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_36_estrategias_para_o_cuidado_da_pessoa_com_doenca_cronica_diabetes_mellituspdf" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_37_estrategias_para_cuidado_da_pessoa_com_doenca_cronica_hipertensao_arterial_sistemicapdf" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_38_estrategias_para_cuidado_da_pessoa_com_doenca_obesidadepdf" + ".pdf",
    "docs/cadernos_de_atencao_basica_no_40_estrategias_para_o_cuidado_da_pessoa_com_doenca_cronica_o_cuidado_da_pessoa_tabagistapdf" + ".pdf"
]
# Sou calouro, preciso fazer matrícula?
# Loading and splitting the documents from multiple PDF files
docs = load_pdf_data(file_paths=pdf_files)
documents = split_docs(documents=docs)
# Creating vectorstore
vectorstore = create_embeddings(documents, embed)

# qa = extract_qa.qaExtractor("ground_truth.txt", "perguntas.txt")
# questions = qa.get_questions()
# ground_truth = qa.get_answers()
# # Converting vectorstore to a retriever
# # search_type= similarity (uses l2 (Euclidian Distance) as default)) search_kwargs = k: 3 (take the top 3 results of the similarity search)
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# # Creating the prompt from the template
# prompt = PromptTemplate.from_template(template)
# count = 0
# dicio = {}
# # print(get_response(retriever, "Quem é Vanessa Oliveira e qual é a sua relação com Diego Madureire no contexto da Universidade de Brasília?", template, llm))
# print("Respondendo questões...")
# for entrada in questions:
#     answer, contexto = get_response(retriever, entrada, template, llm)
#     dicio.update({count : [{"question" : entrada, "answer" : answer, "context": contexto, "ground_truth": ground_truth[count]}]})
#     count += 1
#     # results = vectorstore.similarity_search(query, k=3)
#     # print(f"Retrieved {len(results)} results for the query:")
#     # for i, result in enumerate(results):
#     #     print(f"Result {i+1}:")
#     #     print(f"Content: {result.page_content}")
# file_name = "qa.json"
# with open(file_name, "w", encoding="utf-8") as json_file:
#     json.dump(dicio, json_file, indent=4, ensure_ascii=False)
# print(f"JSON data has been saved to {file_name}")
# # read json file
# with open(file_name, "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)