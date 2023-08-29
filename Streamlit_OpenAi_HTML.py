import os

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredHTMLLoader



import streamlit as st


os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.html'



def process_doc(
        path: str = "C:\\Users\\Mateo\\Downloads\\Documento-de-examen-Grupo1.html",
        is_local: bool = False,
        question: str = 'Quiénes son los autores del pdf?'
):

    _, loader = os.system(f'curl -o {default_doc_name} {path}'), UnstructuredHTMLLoader(f"./{default_doc_name}") if not is_local \
        else UnstructuredHTMLLoader(path)
    doc = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=3500)
    texts = text_splitter.split_documents(doc)
    #print(texts[0].page_content)

    print(texts[-1])

    db = Chroma.from_documents(texts, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.1), chain_type='stuff', retriever=db.as_retriever())
    st.write(qa.run(question))
    print( "\n*** LA PREGUNTA ES:"+question, "\n****"+qa.run(question))


def client():
    st.title('Manage LLM with LangChain')
    uploader = st.file_uploader('Upload html', type='html')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('HTML saved!!')

    question = st.text_input('Generar un resumen de 20 palabras sobre el pdf',
                             placeholder='Give response about your PDF', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default PDF')
            process_doc()


if __name__ == '__main__':
   client()
   #process_doc()