import streamlit as st
from pathlib import Path
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

st.set_page_config(
    page_title="Document GPT",
    page_icon="🐼",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    if Path(f"index/{file.name}").exists():
        vectorstore = FAISS.load_local(f"index/{file.name}", embeddings)

    else:
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n\n",
            chunk_size=600,
            chunk_overlap=150,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        vectorstore.save_local(f"index/{file.name}")

    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=600,
    memory_key="rag_chat_history",
    return_messages=True,
)


def load_memory(_):
    return memory.load_memory_variables({})["rag_chat_history"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
     
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="rag_chat_history"),
        (
            "human",
            "{question}",
        ),
    ]
)

st.title("Document GPT")

st.markdown(
    """
    Welcome to Document GPT!
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a file",
        type=["pdf", "txt", "docx", "xlsx"],
    )

if file:
    retriever = embed_file(file)
    send_message(
        "I'm ready! Ask away!",
        "ai",
        save=False,
    )
    paint_history()
    message = st.chat_input("Ask anything about your file")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "rag_chat_history": RunnableLambda(load_memory),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
            memory.save_context({"input": message}, {"output": response.content})

else:
    st.session_state["messages"] = []
