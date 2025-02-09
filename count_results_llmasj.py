import json
import sys


file_name = str(sys.argv[1])
with open(file_name) as f:
    data = json.load(f)
metrics = list(data[0].keys())
dictionary = dict()
for i in range(len(data)):
    for metric in metrics:
        try:
            if data[i][metric]['Winner'] == 'Answer 2':
                try:
                    dictionary[metric] += 1
                except KeyError:
                    dictionary.update({metric: 1})
        except KeyError:
            continue
print(f"Total: {len(data)}")
print(dictionary)