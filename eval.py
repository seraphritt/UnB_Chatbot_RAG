import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_entity_recall,
    context_recall,
    context_precision,
    answer_relevancy,
    answer_similarity
)
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={
            'normalize_embeddings': normalize_embedding
        }
    )


data_samples = {
    "question": ["O que é a SAA?"],
    "answer":  ['A SAA (Secretaria de Administração Acadêmica) é uma das principais secretarias da Universidade de Brasília, responsável pela gestão dos estudantes e pela expedição de documentos como certificados e diplomas. Ela está localizada em diferentes postos avançados ao longo do campus, incluindo o Posto Avançado da SAA no prédio da Reitoria, onde você pode encontrar a equipe responsável pela solenidade de outorga de grau e pelo envio de documentos. Além disso, a SAA é responsável por garantir a articulação entre o ensino, a pesquisa e a extensão na Universidade de Brasília, promovendo a formação integral e cidadã dos estudantes.'],
    "contexts": [['A Secretaria de Administração Acadêmica é responsável pelo registro dos estudantes e pela expedição de documentos como certificados e diplomas. Para atender melhor os estudantes, a SAA tem postos próximos às unidades acadêmicas. No anexo I você encontra os endereços e telefones de contato dos postos avançados do SAA.']],
    "ground_truth": ['A SAA, ou Secretaria de Administração Acadêmica, é responsável pelo registro dos estudantes e pela expedição de documentos como certificados e diplomas. Ela oferece suporte aos estudantes através de postos próximos às unidades acadêmicas da Universidade de Brasília (UnB). A SAA também é o órgão ao qual os estudantes devem se dirigir para obter históricos escolares atualizados, declarações de vínculo e atestados de matrícula, além de coordenar processos importantes como mudança de curso e dupla diplomação']
}
llm = Ollama(model="llama2", temperature=0.1)
embed = load_embedding_model(model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
dataset = Dataset.from_dict(data_samples)
print(json.dumps(dataset.to_dict(), indent=4, ensure_ascii=False))
result = evaluate(llm=llm, embeddings=embed, dataset=dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_entity_recall,
        context_recall,
        context_precision,
        answer_relevancy,
        answer_similarity
    ],
)
print(result)
