import os

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import PyPDF2
import streamlit as st


os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.pdf'
union1 = 'doc1.pdf'
union2 = 'doc.pdf'
default_doc_name1 = 'pdfCOMBINADO.pdf'
pdf_nombres = []


pdf_url_path= ['doc1.pdf','doc.pdf']
salidaPdf = 'pdfCOMBINADO.pdf'

def combine_pdf(pdf_paths, salida_path):
    pdf_merger = PyPDF2.PdfMerger()
    for pdf_path in pdf_paths:
        pdf_merger.append(pdf_path)

    with open(salida_path, 'wb') as salida_pdf:
        pdf_merger.write(salida_pdf)

def process_doc(

        path2: str = 'pdfCOMBINADO.pdf',
        is_local: bool = False,
        question: str = 'Me puedes dar un resumen corto de 10 lineas dos temas: birds y encoder? '
):


    _, loader = os.system(f'curl -o {default_doc_name1} {path2}'), PyPDFLoader(f"./{default_doc_name1}") if not is_local \
        else PyPDFLoader(path2)


    doc2 = loader.load_and_split()


    print(doc2[-1])


    db = Chroma.from_documents(doc2, embedding=OpenAIEmbeddings())


    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())
    st.write(qa.run(question))
    print( "\n*** LA PREGUNTA ES:"+question, "\n****"+qa.run(question))


def client():

    st.title('UNIR PDFs y Manage LLM with LangChain')
    num_pdfs = st.number_input('Número de pdfs para unir:', min_value=1, step=1)
    for i in range(num_pdfs):
        pdf_uploader = st.file_uploader(f'Suba el PDF {i+1}', type='pdf')
        if pdf_uploader:
            pdf_nombre = f'doc{i+1}.pdf'
            pdf_nombres.append(pdf_nombre)
            with open(f'./{pdf_nombre}','wb') as f:
                f.write(pdf_uploader.getbuffer())
            st.success(f'PDF {pdf_nombre} guardado!')

    if num_pdfs >0:
        question = st.text_input('Generar un resumen de 20 palabras sobre el pdf',
                                 placeholder='Give response about your PDF', disabled=False)

        if st.button('Send Question'):
            if pdf_nombres:
                nombre_pdf_combinados = 'pdfCOMBINADOS1.pdf'
                combine_pdf(pdf_nombres, nombre_pdf_combinados)
                process_doc(
                    path2=nombre_pdf_combinados,
                    is_local=True,
                    question=question
                )
            else:
                st.info('Loading default PDF')
                process_doc()


if __name__ == '__main__':

   client()

