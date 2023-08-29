from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st

#para csv
os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.csv'

def process_doc(
        path: str = 'C:\\Users\\Mateo\\Desktop\\tabla_Calif.csv',
        is_local: bool = False,
        question: str = 'Cual es la nota de Zachary'
):
    loader = CSVLoader(file_path= path, csv_args={
        'delimiter': ';'
    })

    doc = loader.load_and_split()

    #print(doc[-1])
    db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())
    st.write(qa.run(question))
    print( "\n*** LA PREGUNTA ES:"+question, "\n****"+qa.run(question))


def client():
    st.title('Control CSV de LLM con LangChain')
    uploader = st.file_uploader('Sube tu doc .csv', type='csv')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('CSV guardado!!')

    question = st.text_input('Cuál es la nota de Zachary en Química?',
                             placeholder='Genera tu pregunta para el doc .csv', disabled=not uploader)

    if st.button('Envía la pregunta'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question

            )

        else:
            st.info('Cargando el documento .CSV')
            process_doc()



if __name__ == '__main__':
    #process_doc()
    client()
