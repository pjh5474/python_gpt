import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(
    page_title="Site GPT",
    page_icon="📰",
)


def parse_page(soup):
    main = soup.find("main")
    if main:
        content = main.find("div", class_="DocsContent")
        if content:
            return str(content.get_text()).replace("\n", " ")
        else:
            return "no content"
    return "no main"


@st.cache_data(show_spinner="Loading website...")
def load_website(url, keywords):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    filter_urls = [r"^(.*\/{0}\/).*".format(keyword) for keyword in keywords]
    loader = SitemapLoader(
        url,
        filter_urls=filter_urls,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
    return vector_store.as_retriever()


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


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


def clear_keywords():
    st.session_state["keywords"] = []


st.markdown(
    """
    # SiteGPT
    
    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

if "keywords" not in st.session_state:
    st.session_state["keywords"] = ["ai-gateway", "vectorize", "workers-ai"]

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
    )

    if url:
        if ".xml" not in url:
            st.error("Please write down a sitemap URL.")
        else:
            with st.form("keyword", clear_on_submit=True):
                keyword = st.text_input("Put your searching keyword")
                submit = st.form_submit_button("Add")
                if submit:
                    st.session_state["keywords"].append(keyword)

            if len(st.session_state["keywords"]) > 0:
                st.write("Searching Keywords : ")
                st.markdown(st.session_state["keywords"])
                st.button("Clear keywords", on_click=clear_keywords)
                search = st.button("Start !!")
                if search:
                    st.session_state["messages"] = []
                    try:
                        st.session_state["retriever"] = load_website(
                            url, st.session_state["keywords"]
                        )
                        st.session_state["current_keywords"] = st.session_state[
                            "keywords"
                        ]
                    except:
                        st.info(
                            ":red[There may be NO site urls containing your keywords. Please use another keywords.]"
                        )

            else:
                st.info("Please set keywords you are interested in sitemap")
                search = st.button("Start with whole data of sitemap")
                if search:
                    try:
                        st.session_state["retriever"] = load_website(
                            url, st.session_state["keywords"]
                        )
                        st.session_state["current_keywords"] = url
                    except:
                        st.info(":red[Something went wrong...]")

    api_key = st.text_input("Put Your OpenAI API KEY")

    st.info("This page operates based on the below Github repository.")
    st.link_button(
        label="Github", url="https://github.com/pjh5474/python_gpt/tree/streamlit"
    )


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Select the answers that have the highest score (more helpful) and favor the most recent ones.
            Give sources (url) of the answers as they are, do not change them.
            Do Not give answer's score.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="sitegpt_chat_history"),
        ("human", "{question}"),
    ]
)

if api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        api_key=api_key,
    )

    chat_llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        api_key=api_key,
    )

    memory = ConversationBufferMemory(
        llm=chat_llm,
        max_token_limit=600,
        memory_key="sitegpt_chat_history",
        return_messages=True,
    )


def load_memory():
    return memory.load_memory_variables({})["sitegpt_chat_history"]


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | chat_llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    chat_memory = load_memory()
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "sitegpt_chat_history": chat_memory,
        }
    )


@st.cache_data(show_spinner="Waiting for answer...")
def get_sitegpt_answer(question, url, keywords):
    map_re_rank_chain = (
        {
            "docs": st.session_state["retriever"],
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = map_re_rank_chain.invoke(question)
    return result.content.replace("$", "\$")


if "retriever" in st.session_state:
    if api_key:
        send_message(
            f"I'm ready! Ask about {st.session_state['current_keywords']}",
            "ai",
            save=False,
        )
        paint_history()
        user_input = st.chat_input("Ask a question to the website.")

        if user_input:
            send_message(user_input, "human")
            with st.chat_message("ai"):
                gpt_answer = get_sitegpt_answer(
                    question=user_input, url=url, keywords=st.session_state["keywords"]
                )
                save_message(gpt_answer, "ai")
                memory.save_context({"input": user_input}, {"output": gpt_answer})
    else:
        st.info("Please input your API KEY")
else:
    st.session_state["messages"] = []
    st.info("Please press 'Start' button on sidebar.")
