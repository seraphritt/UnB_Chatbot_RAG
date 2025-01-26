import json


class qaExtractor:
    def __init__(self, filename_answer, filename_question):
        self.filename_answer = filename_answer
        self.filename_question = filename_question

    def get_first(self):
        self._answer = []
        with open(self.filename_answer, "r") as respostas:
            for line in respostas:
                if len(line) != 1:
                    self._answer.append(line.strip())
        return self._answer

    def get_second(self):
        self._question = []
        with open(self.filename_question, "r") as respostas:
            for line in respostas:
                if len(line) != 1:
                    self._question.append(line.strip())
        return self._question
    

# file_name = "qa.json"
# with open(file_name, "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)
# print(data["1"][0]["answer"])
