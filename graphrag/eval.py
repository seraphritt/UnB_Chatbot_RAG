import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from matplotlib.backends.backend_pdf import PdfPages
from datasets import Dataset
from ragas import evaluate
import ragas
from ragas.metrics import (
    faithfulness,
    context_entity_recall,
    context_recall,
    context_precision,
    answer_relevancy,
    answer_similarity,
    answer_correctness
)
import pandas as pd
import matplotlib.pyplot as plt
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import FastEmbedEmbeddings

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={
            'normalize_embeddings': normalize_embedding
        }
    )
file_name = "qa.json"
with open(file_name, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

questions = [
    (data[str(x)][0]["question"] if isinstance(data[str(x)], list) else "")
    if str(x) in data else ""
    for x in range(311)
]

answers = [
    (data[str(x)][0]["answer"] if isinstance(data[str(x)], list) else "")
    if str(x) in data else ""
    for x in range(311)
]

ground_truths = [
    (data[str(x)][0]["ground_truth"] if isinstance(data[str(x)], list) else "")
    if str(x) in data else ""
    for x in range(311)
]

contexts = [
    ([data[str(x)][0]["context"]] if isinstance(data[str(x)], list) else [""])
    if str(x) in data else [""]
    for x in range(311)
]

data_samples = {
    "question": questions,
    "answer":  answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}

models = ["llama3.1:latest"]
for model in models:
    # file_name = f"qa_{model}.json"
    # with open(file_name, "r", encoding="utf-8") as json_file:
    #     data = json.load(json_file)
    # questions = [data[str(x)][0]["question"] for x in range(100)]
    # answers = [data[str(x)][0]["answer"] for x in range(100)]
    # ground_truths = [data[str(x)][0]["ground_truth"] for x in range(100)]
    # contexts = [[data[str(x)][0]["context"]] for x in range(100)]
    # data_samples = {
    #     "question": questions,
    #     "answer":  answers,
    #     "contexts": contexts,
    #     "ground_truth": ground_truths,
    # }
    model_name = model
    llm = LangChainLLMWrapper(Ollama(model=model_name, temperature=0.1))
    embed = LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large'))
    run_config = ragas.RunConfig(timeout=180, max_retries=10, max_wait=60)
    dataset = Dataset.from_dict(data_samples)
    result = evaluate(llm=llm, embeddings=embed, dataset=dataset, metrics=[
            context_precision,
            answer_relevancy,
            context_recall,
            faithfulness,
            answer_similarity,
            context_entity_recall,
            answer_correctness,
    ], callbacks  = None, run_config=run_config, raise_exceptions=False,
    )

    print(result)
    df = result.to_pandas()
    csv_file_name = f"evaluation_results_GRAPH_{model_name}.csv"
    df.to_csv(csv_file_name, index=False, encoding='utf-8')
    print(df.keys())
    categories = ['context_precision', 'answer_relevancy', 'context_recall', 'faithfulness', 'semantic_similarity', 'context_entity_recall', 'answer_correctness']
    data = [df[category].dropna() for category in categories]  # Drop NaN 
    plt.figure(figsize=(15, 6))
    plt.boxplot(data, vert=True, patch_artist=True, tick_labels=['Context Precision', 'Answer Relevancy', 'Context Recall', 
                                                            'Faithfulness', 'Answer Similarity', 'Context Entity Recall', 'Answer Correctness'])
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    plt.title(f'Distribução das Métricas (RAGAS) no modelo {model_name}')

    with PdfPages(f'boxplot_graph_{model_name}.pdf') as pdf:
        pdf.savefig()
        plt.close()
    print(f"Documento evaluation_results_{model_name}.csv e boxplot (boxplot_graph_{model_name}_teste.pdf) criados")

