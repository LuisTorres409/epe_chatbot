import os
import uuid
import textwrap
import base64
from io import BytesIO
import torch
import fitz
import streamlit as st
import boto3
import time

from PyPDF2 import PdfReader
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, ServiceContext, VectorStoreIndex, download_loader
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredAPIFileIOLoader, PyMuPDFLoader
from langchain.chat_models import ChatOpenAI
from templates.file_uploader_label import hide_label




## Fazendo o embeddings dos livros
@st.cache_resource(show_spinner=False)
def init_connections_and_databases():
    

    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an assistant to the company EPE, it's a brazilian public company that works with energy. You are responsible for the company's chatbot and you will be talking to company members and assisting with knowlodge of EPE database.Answer all questios in portuguese", max_tokens=2000)
    #llm = ChatOpenAI()

    embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    #embeddings = OpenAIEmbeddings()

    pc = Pinecone()
    pinecone_index = pc.Index("epe-docs")
 
    vectorstore = PineconeVectorStore(pinecone_index=pinecone_index)

    s3 = boto3.client("s3")
    bucket_name = "epe-pdfs"

    Settings.embed_model = embeddings
    Settings.batch_size = 512
    Settings.llm = llm
    index  = VectorStoreIndex.from_vector_store(vectorstore)

    #memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

    chat_engine = index.as_chat_engine(chat_mode="condense_question",
                                       memory=st.session_state.chat_memory,
                                       llm=llm,
                                       context_prompt=("You are an assistant to the company EPE, it's a brazilian public company that works with energy. You are responsible for the company's chatbot and you will be talking to company members and assisting with knowlodge of EPE database.Answer all questios in portuguese"),
                                       verbose=True)
    
    retriever = VectorIndexRetriever(index=index,similarity_top_k=5,verbose=True)

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(retriever=retriever,node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],response_synthesizer=response_synthesizer)

    #query_engine = index.as_query_engine()

    return index , s3 , bucket_name , chat_engine , query_engine


def check_file_exists_in_s3(bucket_name, s3_object_key):
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_object_key)
        return True  # O arquivo existe no S3
    except:
        return False  # O arquivo não existe no S3
    
def download_from_s3(file_name):
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    content = response["Body"].read()
    return content

def preview_pdf(file_bytes):
    encoded_pdf = base64.b64encode(file_bytes).decode('utf-8')
    #return f'<embed src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="500" type="application/pdf">'
    pdf_display =  f"""<embed
    class="pdfobject"
    type="application/pdf"
    title="Embedded PDF"
    src="data:application/pdf;base64,{encoded_pdf}"
    style="overflow: auto; width: 100%; height: 100%;">"""
    return pdf_display

def process_documents(pdf_docs):
    pdfs = []
    for pdf in pdf_docs:
        pdf_bytes = pdf.read()
        s3_object_key = pdf.name
        # Fazer upload do arquivo para o S3
        if not check_file_exists_in_s3(bucket_name,pdf.name):
            pdf_read = PdfReader(pdf)
            for page in pdf_read.pages:
                text = page.extract_text()
                Settings.chunk_size = 2048
                Settings.chunk_overlap = 400
                chunk_document = Document(
                    id=str(uuid.uuid4()),
                    text=text,
                    metadata={'file_name':pdf.name,'page':pdf_read.get_page_number(page),'text':text}
                )
                pdfs.append(chunk_document)
        else:
            error_message = f'{pdf.name} já existe no banco de dados'
            return False , error_message
    
    try:
        for uploaded_pdf in pdfs:
            index.insert(uploaded_pdf)

        s3.upload_fileobj(BytesIO(pdf_bytes), bucket_name, s3_object_key)
        return True , None
    except Exception as e:
        return False, f"Erro ao processar os documentos: {str(e)}"


st.set_page_config(page_title="EPE Chatbot", layout="centered", initial_sidebar_state="expanded", menu_items=None)
st.title('EPE Chatbot')

if "chat_memory" not in st.session_state.keys():
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

index , s3 , bucket_name , chat_engine , query_engine= init_connections_and_databases()
st.sidebar.image('assets/epe_logo.png',width=120)
with st.sidebar:


    tab1 , tab2 , tab3 = st.tabs(["Configurações","Documentos",'Login'])

    with tab1:
        st.subheader("Configurações do Chatbot")
        st.write("Aqui você pode configurar as opções do chatbot")
        retornar_documento = st.checkbox('Retornar a Documentação')
        gerar_documento = st.checkbox('Gerar Documentação')
        st.subheader("Tipo de engine")

        st.write("Chat engine é recomandado para tarefas de conversação, enquanto a query engine é recomendada para tarefas de busca de documentos.")
        
        engine_option = st.radio("Escolha o tipo de engine",["Chat Engine","Query Engine"],index=0,key="engine_type")

    with tab2:
        st.subheader("Seus documentos")
        # Esse hide label foi utilizado para esconder o label do file_uploader que estava em inglês e mudar para pt-br via css, não é uma solução real
        # mas foi a única forma que encontrei de fazer isso.
        st.markdown(hide_label, unsafe_allow_html=True)
        docs = st.file_uploader(label = "Faça upload dos documentos para o chatbot", accept_multiple_files=True,help="Insira aquivos válidos")
        if st.button("Processar documentos"):
            if docs == []:
                st.error("Nenhum documento foi enviado")
            else:
                with st.spinner("Processando"):
                    status,error_message = process_documents(docs)
                    if status:
                        st.success("Documentos processados com sucesso")
                    else:
                        st.error(error_message)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant","avatar":"assets/epe_logo.png", "content": "Olá! Sou o chatbot da EPE, faça perguntas referentes a informações da empresa e sobre documentos que estão na nossa base de dados.Caso deseje enviar um documento para ser analisado, utilize a barra lateral.",'has_file':False}
    ]

if "query_engine" not in st.session_state.keys(): # Initialize the query engine
    st.session_state.query_engine = query_engine

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = chat_engine

if "chat_memory" not in st.session_state.keys():
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

if prompt := st.chat_input("Faça uma pergunta"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt , 'has_file': False})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"],unsafe_allow_html=True)
        if message["has_file"]:
            with st.expander('Documento'):
                #st.write(message["file"], unsafe_allow_html=True)
                doc_link = f'https://epe-pdfs.s3.sa-east-1.amazonaws.com/{message["file"].replace(" ","+")}'
                st.markdown(f'**{message["file"]}**')
                st.link_button('Download',doc_link)
                #st.image(message["img"],use_column_width=True)
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        
        message_placeholder = st.empty()
        message_placeholder.text("Estou processando a sua pergunta...")
        full_response = ''


        #response = st.session_state.query_engine.query(prompt)
        if engine_option == "Chat Engine":
            response = st.session_state.chat_engine.chat(prompt)
        elif engine_option == "Query Engine":
            response = st.session_state.query_engine.query(prompt)
        if retornar_documento:
            try:
                caminho_arquivo = response.source_nodes[0].metadata['file_name']
                nome_arquivo = os.path.basename(caminho_arquivo)
                file_bytes = download_from_s3(nome_arquivo)
                doc_preview = preview_pdf(file_bytes)
                doc_image = fitz.open(stream=file_bytes, filetype='pdf')
                page_image = doc_image.load_page(0)
                pix = page_image.get_pixmap(dpi=300)  # Scale up the image resolution
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                st.write(response)
                response_message = (response.response + f'\n\n Aqui está o documento relacionado a sua pergunta: ')

                for chunk in response_message.split():
                    
                    full_response += chunk + " "
                    time.sleep(0.025)
                    message_placeholder.markdown(full_response + " ")

                doc_link = f'https://epe-pdfs.s3.sa-east-1.amazonaws.com/{nome_arquivo.replace(" ","+")}'
                st.markdown(f'**{nome_arquivo}**')
                st.link_button('Download',doc_link)
                with st.expander('Pré vizualizar documento'):
                    st.image(img,use_column_width=True)
                st.session_state.messages.append({"role": "assistant", "content": response_message, "file": nome_arquivo , "has_file": True , "img": img})
            
            except Exception as e:
                response_message = response.response
                for chunk in response_message.split():
                    
                    full_response += chunk + " "
                    time.sleep(0.025)
                    message_placeholder.markdown(full_response + " ")
                st.error(f"Ocorreu um erro {e}")
                st.session_state.messages.append({"role": "assistant", "content": response_message, "has_file": False})
        else:
            response_message = response.response
            for chunk in response_message.split():
                
                full_response += chunk + " "
                time.sleep(0.025)
                message_placeholder.markdown(full_response + " ")

            st.session_state.messages.append({"role": "assistant", "content": response_message, "has_file": False})

