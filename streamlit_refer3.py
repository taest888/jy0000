import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
        page_title="대화형 ai를 활용한 업무매뉴얼",
        page_icon=":droplet:",
        layout="wide"
    )

    st.markdown(
        """
        <style>
            .css-18e3th9 {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }
            .st-bb {
                background-color: #e0f7fa;
            }
            .st-df {
                background-color: #0288d1;
            }
            .st-ag {
                color: #ffffff;
            }
            .css-1kyxreq {
                background-color: #ff9800;
            }
            .css-2trqyj {
                background-color: #0288d1;
            }
            .css-1q1n0ol {
                color: #0288d1;
            }
            .css-1m4d6r4 {
                font-size: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("_AI 정수장  :blue[Q&A ChatBot]_ :droplet:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        st.markdown("<h2 style='color: #0288d1;'>Upload Files</h2>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("매뉴얼 업로드", type=['pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Click Me")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        st.write("Processing started...")  # 디버깅 메시지
        files_text = get_text(uploaded_files)
        st.write("Files text obtained")  # 디버깅 메시지
        text_chunks = get_text_chunks(files_text)
        st.write("Text chunks obtained")  # 디버깅 메시지
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vector store created")  # 디버깅 메시지

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        st.write("Conversation chain initialized")  # 디버깅 메시지

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 정수장에 관해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        if st.session_state.conversation is None:
            st.error("Conversation chain is not initialized. Please process the files first.")
            st.stop()

        chain = st.session_state.conversation
        st.write("Conversation chain ready")  # 디버깅 메시지

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    st.write("LLM initialized")  # 디버깅 메시지
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    st.write("Conversation chain created")  # 디버깅 메시지
    return conversation_chain

if __name__ == '__main__':
    main()
