import requests
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI  # OpenAI 패키지 추가
from langchain.docstore.document import Document  # Document 클래스 임포트

# 환경 변수 로드
load_dotenv()

# Chroma DB가 저장된 디렉토리 경로
persist_directory = './db/chromadb'

# 제목 및 기본 설명
st.set_page_config(
    page_title="News-Chat",
    layout="wide",
)
st.title("News-Chat")
st.write("---")

# Session state 초기화
if 'news_data_collected' not in st.session_state:
    st.session_state.news_data_collected = False
    st.session_state.chromadb = None
    st.session_state.news_summary = ""
    st.session_state.chat = []
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    st.session_state.keyword = ""
    st.session_state.selected_model = "gpt-3.5-turbo"
    st.session_state.expander_expanded = True  # expander 상태 초기화

# 네이버 뉴스 API를 사용하여 뉴스 데이터 수집
def fetch_news(keyword, start=1):
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
    }
    params = {
        "query": keyword,
        "display": 10,  # 최대 100개까지 요청 가능
        "start": start,
        "sort": "date"
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# 뉴스 데이터 전처리
def preprocess_news(data, keyword):
    articles = data['items']
    cleaned_articles = []
    for article in articles:
        if keyword in article['title'] or keyword in article['description']:  # 제목 또는 설명에 키워드가 있는지 확인
            cleaned_articles.append({
                "title": article['title'],
                "description": article['description'],
                "pub_date": article['pubDate']
            })
    return cleaned_articles

# 뉴스 데이터를 문서로 변환
def news_to_documents(news_data):
    texts = []
    for article in news_data:
        content = f"{article['title']}\n{article['description']}\n{article['pub_date']}"
        texts.append(Document(page_content=content))
    return texts

# OpenAI 클라이언트 초기화
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# 입력 텍스트에서 키워드 추출
def extract_keyword(text):
    prompt = [{"role": "user", "content": f"다음 문장에서 주요 뉴스 검색 키워드를 추출해줘. '가격','검색', '인상', '인하' 같은 일반적인 단어는 제외하고 딱 하나만(공백없이) 추출해줘.: {text}"}]
    response = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=prompt,
        temperature=0
    )
    keyword = response.choices[0].message.content.strip()
    return keyword

def summarize_news(articles, keyword):
    content = "\n\n".join([f"{article['title']}\n{article['description']}" for article in articles])
    prompt = [{"role": "user", "content": f"'{keyword}' 키워드와 관련된 다음 뉴스 기사들을 요약해줘:\n\n{content}"}]
    response = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=prompt,
        temperature=0.5  # 온도 조정
    )
    summary = response.choices[0].message.content
    return summary

def search_in_chromadb(query):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_embedding = embeddings_model.embed_query(query)
    search_results = st.session_state.chromadb.similarity_search(query, k=5)
    return search_results

# 사이드바 생성
with st.sidebar:
    # Open AI API 키 입력받기
    openai_api_key = st.text_input(label="OPENAI API 키", placeholder="Enter Your API Key", value="", type="password")
    client.api_key = openai_api_key

    st.markdown("---")

    # GPT 모델을 선택하기 위한 라디오 버튼 생성
    model = st.radio(label="GPT 모델", options=["gpt-4", "gpt-3.5-turbo"])
    st.session_state.selected_model = model

    st.markdown("---")

    # 리셋 버튼 생성
    if st.button(label="초기화"):
        # 리셋 코드
        st.session_state.news_data_collected = False
        st.session_state.chromadb = None
        st.session_state.news_summary = ""
        st.session_state.chat = []
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.session_state.keyword = ""
        st.query_params = {"reset": "true"}

# 메인 영역
if not st.session_state.news_data_collected:
    with st.expander("뉴스 데이터 수집", expanded=st.session_state.expander_expanded):
        keyword_input = st.text_input("키워드를 입력하세요", key="news_keyword")
        if st.button('데이터 수집 시작', key="fetch_data"):
            if keyword_input:
                try:
                    progress_bar = st.progress(0)
                    st.write("키워드에 대한 뉴스 데이터 수집 중...")
                    
                    # 입력 텍스트에서 키워드 추출
                    keyword = extract_keyword(keyword_input)
                    st.write(f"추출된 키워드: {keyword}")
                    
                    news_data = fetch_news(keyword)
                    progress_bar.progress(20)

                    articles = preprocess_news(news_data, keyword)
                    st.write("뉴스 데이터 수집 완료.")
                    progress_bar.progress(40)

                    # Split
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=50,
                        length_function=len,
                        is_separator_regex=False,
                    )
                    texts = text_splitter.split_documents(news_to_documents(articles))
                    st.write("뉴스 데이터를 텍스트 청크로 분할 완료.")
                    progress_bar.progress(60)

                    # Embedding
                    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                    embeddings = []
                    for i, text in enumerate(texts):
                        embedding = embeddings_model.embed_documents([text.page_content])
                        embeddings.append(embedding)
                        progress = 60 + int((i + 1) / len(texts) * 20)
                        progress_bar.progress(progress)
                    st.write(f"텍스트를 임베딩으로 변환 완료. 임베딩 개수: {len(embeddings)}")

                    # Load it into Chroma
                    if not os.path.exists(persist_directory):
                        chromadb = Chroma.from_documents(
                            texts,
                            embeddings_model,
                            collection_name='news',
                            persist_directory=persist_directory,
                        )
                        st.write("Chroma 데이터베이스 생성 완료.")
                    else:
                        chromadb = Chroma(
                            persist_directory=persist_directory,
                            embedding_function=embeddings_model,
                            collection_name='news'
                        )
                        chromadb.add_documents(texts)
                        st.write("기존 Chroma 데이터베이스에 새 데이터 추가 완료.")
                    progress_bar.progress(100)

                    st.session_state.news_data_collected = True
                    st.session_state.chromadb = chromadb
                    st.session_state.keyword = keyword

                    # 뉴스 요약 요청 및 저장
                    st.session_state.news_summary = summarize_news(articles, keyword)

                    # 데이터 수집 후 expander 닫기
                    st.session_state.expander_expanded = False
                    st.experimental_set_query_params(expander_expanded="false")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
if st.session_state.news_data_collected:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"'{st.session_state.keyword}' 키워드 관련 오늘의 뉴스 요약")
        st.write(st.session_state.news_summary)

    with col2:
        st.subheader("뉴스에 대해 질문해보세요!!")
        question = st.text_input('질문을 입력하세요', key="user_question")

        if st.button('질문하기', key="ask_advanced_question"):
            with st.spinner('Wait for it...'):
                try:
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # 임베딩 기반 검색
                    search_results = search_in_chromadb(question)
                    search_snippets = "\n\n".join([result.page_content for result in search_results])
                    
                    # GPT 모델에게 질문과 검색 결과를 기반으로 답변 생성 요청
                    prompt = st.session_state.messages + [{"role": "system", "content": f"다음은 검색 결과입니다:\n\n{search_snippets}"}]
                    response = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=prompt,
                        temperature=0.5
                    )
                    answer = response.choices[0].message.content

                    # GPT 모델에 넣을 프롬프트를 위해 답변 내용 저장
                    st.session_state.messages.append({"role": "system", "content": answer})

                    # 채팅 시각화를 위한 질문과 답변 내용 저장
                    now = datetime.now().strftime("%H:%M")
                    st.session_state.chat.append(("user", now, question))
                    st.session_state.chat.append(("bot", now, answer))

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # 채팅 형식으로 시각화 하기
    with col2:
        for sender, time, message in reversed(st.session_state.chat):  # 최신 메시지가 상단에 오도록 역순으로 출력
            if sender == "user":
                st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                st.write("")
            else:
                st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                st.write("")

