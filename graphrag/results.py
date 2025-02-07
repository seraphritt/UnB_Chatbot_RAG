import pandas as pd

file_name = "evaluation_results_GRAPH_mistral:latest.csv"
df = pd.read_csv(file_name)
df.fillna(0, inplace=True)
print('context precision mean')
print(sum(list(df['context_precision'])) / df.shape[0])

print('answer relevancy mean')
print(sum(list(df['answer_relevancy'])) / df.shape[0])

print('context recall mean')
print(sum(list(df['context_recall'])) / df.shape[0])

print('faithfulness mean')
print(sum(list(df['faithfulness'])) / df.shape[0])
try:
    df['answer_similarity']
    print('answer similarity mean')
    print(sum(list(df['answer_similarity'])) / df.shape[0])
except KeyError:
    print('semantic similarity mean')
    print(sum(list(df['semantic_similarity'])) / df.shape[0])

print('context entity recall relevancy mean')
print(sum(list(df['context_entity_recall'])) / df.shape[0])

print('answer correctness mean')
print(sum(list(df['answer_correctness'])) / df.shape[0])
