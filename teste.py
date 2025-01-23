import json
file_name = "DataSetDictPerguntasRespostas.json"
with open(file_name, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
print(data.keys())