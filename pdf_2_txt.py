from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf_data(file_paths):
    all_docs = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


file_name_list = ["caderno_34", "cadernos_de_atencao_basica_no_13_canceres_do_colo_do_utero_e_da_mamapdf", "cadernos_de_atencao_basica_no_19_envelhecimento_e_saude_da_pessoa_idosapdf", 
                  "cadernos_de_atencao_basica_no_20_carencias_de_micronutrientespdf", "cadernos_de_atencao_basica_no_23_saude_da_crianca_aleitamento_materno_e_alimentacao_complementarpdf", 
                  "cadernos_de_atencao_basica_no_29_rastreamentopdf", "cadernos_de_atencao_basica_no_32_atencao_ao_pre_natal_de_baixo_riscopdf", 
                  "cadernos_de_atencao_basica_no_33_saude_da_crianca_crescimento_e_desenvolvimentopdf", "cadernos_de_atencao_basica_no_35_estrategias_para_o_cuidado_da_pessoa_com_doenca_cronicapdf", 
                  "cadernos_de_atencao_basica_no_36_estrategias_para_o_cuidado_da_pessoa_com_doenca_cronica_diabetes_mellituspdf", "cadernos_de_atencao_basica_no_37_estrategias_para_cuidado_da_pessoa_com_doenca_cronica_hipertensao_arterial_sistemicapdf", 
                  "cadernos_de_atencao_basica_no_38_estrategias_para_cuidado_da_pessoa_com_doenca_obesidadepdf", "cadernos_de_atencao_basica_no_40_estrategias_para_o_cuidado_da_pessoa_com_doenca_cronica_o_cuidado_da_pessoa_tabagistapdf"]
for file_name in file_name_list:
    content = load_pdf_data([f"docs/{file_name}" + ".pdf"])
    content_vector = []
    for each in range(len(content)):
        content_vector.append(content[each].page_content)
    with open(file_name + ".txt", "w", encoding="utf-8") as file:
        for line in content_vector:
            file.write(line + "\n")

