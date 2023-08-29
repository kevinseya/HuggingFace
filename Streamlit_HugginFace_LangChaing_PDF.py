import tracemalloc

import torch
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import streamlit as st
from translate import Translator
#ApiKey HF
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
#RutaPDF
default_doc_name = 'doc.pdf'

def process_doc(
        path: str = "http://www.scielo.org.pe/pdf/rmh/v31n2/1729-214X-rmh-31-02-125.pdf",
        is_local: bool = False,
        question: str = '¿De qué trata el documento?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)
    doc = loader.load_and_split()

#SEPARADOR DE DOCUMENTO EN SEGMENTOS DE 500
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(doc)
    print(texts[0].page_content)
#CREACCIÓN DEL EMBEDDING
    embeddings = SentenceTransformerEmbeddings(model_name="MBZUAI/LaMini-T5-738M")
    db = Chroma.from_documents(texts, embeddings)
#PARA HACER CONSULTA EN EL CHROMA
    tracemalloc.start()
    similar_documents = db.search(
        search_type="similarity",
        query="Medicina",
        n_results=2
    )
    print('*****************')
    print(similar_documents)
#UTILIZACIÓN DEL MODELO ALOJADO EN HUGGING FACE
    checkpoint = "MBZUAI/LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype = torch.float32
    )
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer= tokenizer,
        max_length= 256,
        do_sample= True,
        temperature= 0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    retriever = db.as_retriever()
#UTILIZACIÓN DE RETRIEVER PARA PREGUNTA Y RESPUESTA
    qa = RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=retriever
    )
    respuesta=qa(question)
    answer = respuesta['result']
    #Traducimos la respuesta
    #el modelo MBZUAI/LaMini-T5-738M es en ingles
    translator = Translator(to_lang="es")
    answer_trad = translator.translate(answer)
    st.write(answer_trad)
    print("\n*** LA PREGUNTA ES:" + question, "\n****" + answer_trad)

def client():
    st.title('Control PDF de LLM con LangChain')
    uploader = st.file_uploader('Sube tu doc PDF', type='pdf')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('PDF guardado!!')

    question = st.text_input('¿De qué trata el documento?',
                             placeholder='Genera tu pregunta para el doc .pdf', disabled=not uploader)

    if st.button('Envía la pregunta'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question

            )

        else:
            st.info('No cargaste ningún documento, se generará pregunta default con documento default')
            process_doc()



if __name__ == '__main__':
    #process_doc()
    client()