# SQLite 설정
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

import os
import streamlit as st
import tempfile

# Chroma DB가 저장된 디렉토리 경로
persist_directory = '/app/db/chromadb'  # 절대 경로 사용

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("Choose a file")
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    try:
        pages = pdf_to_document(uploaded_file)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 더 작은 크기로 조정
            chunk_overlap=100,  # 더 큰 오버랩 설정
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(pages)

        # Embedding
        embeddings_model = OpenAIEmbeddings()

        # Load into Chroma
        if not os.path.exists(persist_directory):
            chromadb = Chroma.from_documents(
                texts, 
                embeddings_model,
                collection_name='esg',
                persist_directory=persist_directory,
            )
        else:
            chromadb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings_model,
                collection_name='esg'
            )

        # Question
        st.header("PDF에게 질문해보세요!!")
        question = st.text_input('질문을 입력하세요')

        if st.button('질문하기'):
            with st.spinner('Wait for it...'):
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(
                                llm,
                                retriever=chromadb.as_retriever(search_kwargs={"k": 3}),
                                return_source_documents=True
                            )
                result = qa_chain({"query": question})

                # 검색 결과 및 원본 문서 표시
                st.write("검색 결과:")
                st.write(result["result"])

                st.write("원본 문서:")
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

    except Exception as e:
        st.error(f"An error occurred: {e}")
