import os

from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import streamlit as st

os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.pdf'
def process_doc(
        path: str = "C:\\Users\\Mateo\\Desktop\\doc1.pdf",
        is_local: bool = False,
        question: str = 'Quiénes son los autores del pdf?'
):

    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)
    doc = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=3500,
        length_function=len,
    )
    texts = text_splitter.split_documents(doc)
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'gpu'}
    hf_embedding = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs
    )

    db = Chroma.from_documents(texts, embedding=hf_embedding)
    llm = OpenAI(temperature=0.9)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.1), chain_type='stuff', retriever=db.as_retriever())
    respuesta = qa.run(question)
    st.write(qa.run(question))

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

    client()



