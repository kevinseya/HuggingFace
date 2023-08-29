import os

from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

#OPENAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
#Por si ocupo OPENAI
os.environ['OPENAI_API_KEY'] = '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'

default_doc_name = 'doc.pdf'
def process_doc(
        path: str = "http://www.scielo.org.pe/pdf/rmh/v31n2/1729-214X-rmh-31-02-125.pdf",
        is_local: bool = False,
        question: str = '¿Cuál es la estructura viral del SARS-CoV-2?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)
    doc = loader.load_and_split()

    #print(doc[0].page_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=3500)
    texts = text_splitter.split_documents(doc)
    print(texts[0].page_content)

    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(texts, embedding=embeddings)

    #modeloGPT2 de HUGGINFACE
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        pad_token_id=None,
        eos_token_id=50256
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.1), chain_type='stuff', retriever=db.as_retriever())

    print("\n*** LA PREGUNTA ES:" + question, "\n****" + qa.run(question))


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
