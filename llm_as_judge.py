import extract_qa
import pandas as pd
from langchain.llms import Ollama
import re
import json
qa = extract_qa.qaExtractor("perguntas_unb.txt", "")
queries = qa.get_first()
model1 = pd.read_csv('evaluation_results_qwen2.5:latest.csv')
model2 = pd.read_csv('evaluation_results_GRAPH_qwen2.5:latest.csv')
llm = Ollama(model="llama3.1:8b-instruct-q4_K_M", temperature=0.1)
answers1_vectorstore = model1['response']
answers2_graph = model2['response']
lista = []
for i, (query, answer1, answer2) in enumerate(zip(queries, answers1_vectorstore, answers2_graph)):
    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers in Portuguese to the same question in Portuguese based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
    """

    prompt = f"""
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

    Here is the question:
    {query}

    Here are the two answers:

    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format:
    ```json
    {{
        "Comprehensiveness": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Empowerment": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Provide explanation here]"
        }},
        "Overall Winner": {{
            "Winner": "[Answer 1 or Answer 2]",
            "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
        }}
    }}
    ```
    """
    print("PERGUNTA")
    print(query)
    text = llm.invoke(sys_prompt + prompt)
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    print(text)
    if match:
        json_str = match.group(1)
        json_data = json.loads(json_str)
        lista.append(json_data)
    with open("qwen_comp_unb.json", "w", encoding="utf-8") as json_file:
        json.dump(lista, json_file, indent=4, ensure_ascii=False)