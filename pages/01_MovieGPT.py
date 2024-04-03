import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

st.set_page_config(
    page_title="Movie to ICONS",
    page_icon="",
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


def save_message(message, role):
    st.session_state["movie-emoji-messages"].append(
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
    for message in st.session_state["movie-emoji-messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


examples = [
    {
        "movie": "Inception",
        "answer": """
        🧠💭🌀
        """,
    },
    {
        "movie": "The Shawshank Redemption",
        "answer": """
        ⛓️🔨🏃
        """,
    },
    {
        "movie": "Avatar",
        "answer": """
        🌿🔵🌌
        """,
    },
    {
        "movie": "The Dark Knight",
        "answer": """
        🦇🃏🌃
        """,
    },
    {
        "movie": "Forrest Gump",
        "answer": """
        🏃🍫🎈
        """,
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Using only 3 emojis, explain about this movie: {movie}"),
        ("ai", "{answer}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie mania and you are participating in a movie-emoji game. When introducing a movie, You have to sum up the movie in three emojis. Learn 'how to play the game' and participate in the movie-emoji game",
        ),
        ("system", "How to play the game : "),
        few_shot_prompt,
        (
            "system",
            "Let's start movie-emoji game!",
        ),
        MessagesPlaceholder(variable_name="rag_chat_history"),
        (
            "human",
            "{question}",
        ),
    ]
)

st.title("Movie to Icons")

st.markdown(
    """
    Welcome to ICONS!
    """
)

if "movie-emoji-messages" not in st.session_state.keys():
    st.session_state["movie-emoji-messages"] = []

with st.sidebar:
    api_key = st.text_input("Put Your OpenAI API KEY")

    st.info("This page operates based on the below Github repository.")
    st.link_button(
        label="Github", url="https://github.com/pjh5474/python_gpt/tree/streamlit"
    )

if api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        api_key=api_key,
    )

    memory = ConversationBufferMemory(
        llm=llm,
        max_token_limit=600,
        memory_key="rag_chat_history",
        return_messages=True,
    )


def load_memory(_):
    return memory.load_memory_variables({})["rag_chat_history"]


if api_key:
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
    st.info("Please input your OpenAI KEY")
