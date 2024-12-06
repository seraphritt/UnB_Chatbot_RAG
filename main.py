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
    summarization_score,
    answer_relevancy,
    context_recall,
    context_precision,
)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
You are a respectful and honest assistant specialized to answer ONLY about University of Brasília. \
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
    context = retriever.get_relevant_documents(query)[0].page_content
    print("CONTEXTO")
    print(context)
    print("RESPOSTA")
    print(llm.invoke(template.format(context=context, question=query)))
    print("------------------------------------------")

llm = Ollama(model="mistral", temperature=0.2)
embed = load_embedding_model(model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# List of PDF files to be processed
pdf_files = ["manual_para_estudantes_2022.pdf", "guia_calouro_1_2018.pdf",]
# Sou calouro, preciso fazer matrícula?
# Loading and splitting the documents from multiple PDF files
docs = load_pdf_data(file_paths=pdf_files)
documents = split_docs(documents=docs)

# Creating vectorstore
vectorstore = create_embeddings(documents, embed)

# Check the number of vectors stored in the FAISS index
print(f"Number of vectors: {vectorstore.index.ntotal}")

# Test a sample query to verify retrieval
query = "O que é SAA?"
results = vectorstore.similarity_search(query, k=3)
print(results)
print(f"Retrieved {len(results)} results for the query:")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")

# Converting vectorstore to a retriever
# search_type= similarity (uses l2 (Euclidian Distance) as default)) search_kwargs = k: 3 (take the top 3 results of the similarity search)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print(retriever.get_relevant_documents("O que é SAA?"))
# Creating the prompt from the template
prompt = PromptTemplate.from_template(template)
print(prompt)
# Sample data for evaluation
data = {
    "question": ["O que é a SAA?"],
    "answer":  ['A SAA (Secretaria de Administração Acadêmica) é uma das principais secretarias da Universidade de Brasília, responsável pela gestão dos estudantes e pela expedição de documentos como certificados e diplomas. Ela está localizada em diferentes postos avançados ao longo do campus, incluindo o Posto Avançado da SAA no prédio da Reitoria, onde você pode encontrar a equipe responsável pela solenidade de outorga de grau e pelo envio de documentos. Além disso, a SAA é responsável por garantir a articulação entre o ensino, a pesquisa e a extensão na Universidade de Brasília, promovendo a formação integral e cidadã dos estudantes.'],
    "contexts": [['A Secretaria de Administração Acadêmica é responsável pelo registro dos estudantes e pela expedição de documentos como certificados e diplomas. Para atender melhor os estudantes, a SAA tem postos próximos às unidades acadêmicas. No anexo I você encontra os endereços e telefones de contato dos postos avançados do SAA.']],
    "ground_truth": ['A SAA, ou Secretaria de Administração Acadêmica, é responsável pelo registro dos estudantes e pela expedição de documentos como certificados e diplomas. Ela oferece suporte aos estudantes através de postos próximos às unidades acadêmicas da Universidade de Brasília (UnB). A SAA também é o órgão ao qual os estudantes devem se dirigir para obter históricos escolares atualizados, declarações de vínculo e atestados de matrícula, além de coordenar processos importantes como mudança de curso e dupla diplomação']
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Debug: Print dataset structure
print(json.dumps(dataset.to_dict(), indent=4, ensure_ascii=False))

# Run the evaluation
result = evaluate(llm=llm, embeddings=embed, dataset=dataset, metrics=[
       context_precision,
       answer_relevancy,
   ],
)

print(result)

df = result.to_pandas()

print("DATAFRAME \n\n")
print(float(df.loc[0, 'context_precision']))

print(df)

categories = ['Context Precision', 'Answer Relevancy', 'Category C', 'Category D']
values = [round(float(df.loc[0, 'context_precision']), 2), round(float(df.loc[0, 'answer_relevancy']), 2), 0.2, 0.75]

plt.figure(figsize=(10, 6))
plt.bar(categories, values)
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('RAG Metrics')

# Save to PDF
with PdfPages('bar_graph.pdf') as pdf:
    pdf.savefig() 
    plt.close()   

while True:
    entrada = input()
    get_response(retriever, entrada, template, llm)
    print(results)
    results = vectorstore.similarity_search(query, k=3)
    print(f"Retrieved {len(results)} results for the query:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result.page_content}")