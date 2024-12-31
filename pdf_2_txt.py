from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf_data(file_paths):
    all_docs = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

file_name = "manual_estagio_nao_obrigatorio_discentes"
content = load_pdf_data([f"docs/{file_name}" + ".pdf"])
content_vector = []
for each in range(len(content)):
    content_vector.append(content[each].page_content)
with open(file_name + ".txt", "w", encoding="utf-8") as file:
    for line in content_vector:
        file.write(line + "\n")

