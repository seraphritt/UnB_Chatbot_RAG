# teste de levene
import scipy.stats as stats
import pandas as pd
# Aplicando o teste de Levene
evals = pd.DataFrame({
    'context_precision': [0.899, 0.199, 0.379, 0.269],
    'context_recall': [0.390, 0.146, 0.274, 0.270],
    'context_entity_recall': [0.200, 0.094, 0.131, 0.120],
    'faithfullness': [0.020, 0.216, 0.532, 0.393],
    'answer_relevance': [0.280, 0.338, 0.633, 0.384],
    'answer_similarity': [0.886, 0.884, 0.884, 0.887],
    'answer_correctness': [0.009, 0.074, 0.089, 0.164]
})
metrics = list(evals.keys())
# evals é uma lista de dataframes, onde cada dataframe é um experimento, no seu caso modelo

for metric in metrics:
    stat, p_value = stats.levene(*[evals[metric].values])

    # Exibindo os resultados
    print(f"Metrica: {metric}")
    print(f"Estatística de Levene: {stat:.4f}, p-valor: {p_value:.4f}")

    # Interpretando o resultado
    if p_value < 0.05:
        print("As variâncias são diferentes (heterocedasticidade). Use Welch’s ANOVA.")
    else:
        print("As variâncias são iguais (homocedasticidade). Pode usar ANOVA tradicional.")
    print()
