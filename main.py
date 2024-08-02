<<<<<<< HEAD
=======
# SQLite 설정
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

>>>>>>> ee0dd9ac0b2b3af9d5932be19c715fa38691083d
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import os
import streamlit as st
import tempfile
import re

# Chroma DB가 저장된 디렉토리 경로
persist_directory = '/tmp/chromadb'  # 일시적 디렉토리 사용

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

<<<<<<< HEAD
=======
def preprocess_text(text):
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 문장 끝에 마침표 추가
    text = re.sub(r'(?<!\.)\n', '.\n', text)
    return text

# 업로드 되면 동작하는 코드
>>>>>>> ee0dd9ac0b2b3af9d5932be19c715fa38691083d
if uploaded_file is not None:
    try:
        st.write("파일 업로드 완료. PDF 처리 중...")
        pages = pdf_to_document(uploaded_file)
        st.write("PDF를 페이지로 분할 완료.")

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
<<<<<<< HEAD
            chunk_size = 1000,
            chunk_overlap = 50,
            length_function = len,
            is_separator_regex = False,
=======
            chunk_size=500,  # 더 작은 크기로 조정
            chunk_overlap=100,  # 더 큰 오버랩 설정
            length_function=len,
            is_separator_regex=False,
>>>>>>> ee0dd9ac0b2b3af9d5932be19c715fa38691083d
        )
        texts = text_splitter.split_documents(pages)
        st.write("페이지를 텍스트 청크로 분할 완료.")

<<<<<<< HEAD
        # Embedding
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        embeddings = embeddings_model.embed_documents([text.page_content for text in texts])
        st.write(f"텍스트를 임베딩으로 변환 완료. 임베딩 개수: {len(embeddings)}")

        # Load it into Chroma
        if not os.path.exists(persist_directory):
            chromadb = Chroma.from_documents(
                texts,
=======
        # 전처리 적용
        preprocessed_texts = [preprocess_text(text.page_content) for text in texts]

        # Embedding
        embeddings_model = OpenAIEmbeddings()

        # Load into Chroma
        if not os.path.exists(persist_directory):
            chromadb = Chroma.from_documents(
                preprocessed_texts, 
>>>>>>> ee0dd9ac0b2b3af9d5932be19c715fa38691083d
                embeddings_model,
                collection_name='esg',
                persist_directory=persist_directory,
            )
            st.write("Chroma 데이터베이스 생성 완료.")
        else:
            chromadb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings_model,
                collection_name='esg'
            )
            st.write("기존 Chroma 데이터베이스 로드 완료.")

        # Question
        st.header("PDF에게 질문해보세요!!")
        question = st.text_input('질문을 입력하세요')

        if st.button('질문하기'):
            with st.spinner('Wait for it...'):
                st.write("질문 처리 중...")
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=chromadb.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                result = qa_chain({"query": question})
<<<<<<< HEAD
                st.write("질문에 대한 응답 완료.")
                st.write(result["result"])

                # Source documents 확인
                st.write("출처 문서:")
                for doc in result['source_documents']:
                    st.write(doc.page_content)
=======

                # 검색 결과 및 원본 문서 표시
                st.write("검색 결과:")
                st.write(result["result"])

                st.write("원본 문서:")
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

>>>>>>> ee0dd9ac0b2b3af9d5932be19c715fa38691083d
    except Exception as e:
        st.error(f"An error occurred: {e}")
