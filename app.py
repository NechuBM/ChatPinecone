import streamlit as st
import os
from utils import *
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

FILE_LIST = "archivos.txt"
OPENAI_API_KEY = "Añadir OpenAI API Key"

st.set_page_config('preguntaDOC')
st.header("Pregunta a tu PDF")

with st.sidebar:
    archivos = load_name_files(FILE_LIST)
    files_uploaded = st.file_uploader(
        "Carga tu archivo",
        type="pdf",
        accept_multiple_files=True
        )
    
    if st.button('Procesar'):
        for pdf in files_uploaded:
            if pdf is not None and pdf.name not in archivos:
                archivos.append(pdf.name)
                text_to_pinecone(pdf)

        archivos = save_name_files(FILE_LIST, archivos)

    if len(archivos)>0:
        st.write('Archivos Cargados:')
        lista_documentos = st.empty()
        with lista_documentos.container():
            for arch in archivos:
                st.write(arch)
            if st.button('Borrar Documentos'):
                archivos = []
                clean_files(FILE_LIST)
                lista_documentos.empty()


if len(archivos)>0:
    user_question = st.text_input("Pregunta: ")
    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        vstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)

        docs = vstore.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.write(respuesta)