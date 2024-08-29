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
from pandas.plotting import table

def load_pdf_data(file_paths):
    all_docs = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda'},
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
You are an respectful and honest assistant specialized to answer about University of Brasília. \
Always answer in Portuguese from Brazil. \
All your answers from now on must be in Portuguese. \
All your answers must be strictly related to the context passed for you.
If the question is not related to the context of the University of Brasília, you must say that you don't know how to answer. 

### Context:
{context}

### User:
{question}

### Response:
"""

def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def get_response(query, chain):
    response = chain({'query': query})
    print(f"{response['result']}\n\n")

llm = Ollama(model="llama2", temperature=0.1)
embed = load_embedding_model(model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# List of PDF files to be processed
pdf_files = ["guia_calouro_1_2018.pdf","manual_para_estudantes_2022.pdf"]

# Loading and splitting the documents from multiple PDF files
docs = load_pdf_data(file_paths=pdf_files)
documents = split_docs(documents=docs)

# Creating vectorstore
vectorstore = create_embeddings(documents, embed)

# Converting vectorstore to a retriever
retriever = vectorstore.as_retriever()

# Creating the prompt from the template
prompt = PromptTemplate.from_template(template)
print(prompt)

# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)

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
# print(json.dumps(dataset.to_dict(), indent=4, ensure_ascii=False))

# Run the evaluation
#result = evaluate(llm=llm, embeddings=embed, dataset=dataset, metrics=[
#        context_precision,
#        answer_relevancy,
#    ],
#)

# Check and print the results
#print(result)

# Convert results to pandas dataframe
#df = result.to_pandas()

# Debug: Print the dataframe
#print(df)

# Create a matplotlib figure and axis
#fig, ax = plt.subplots(figsize=(10, 4))  # You can adjust the size
#ax.axis('tight')
#ax.axis('off')

# Create a table plot
#tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2] * len(df.columns))

# Save the table as a PDF
#plt.savefig("evaluation_results.pdf")
#plt.close()
while True:
    get_response("\n\n Remember to always answer in Portuguese" + input(), chain)
