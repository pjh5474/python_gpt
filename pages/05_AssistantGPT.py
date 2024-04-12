import pages.assistant.investor_assistant as investor_assistant
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from openai import OpenAI
import time


def make_assistant():
    client = OpenAI(api_key=api_key)
    assistant_id = investor_assistant.make_assistant_id(client)
    st.session_state["client"] = client
    st.session_state["assistant_id"] = assistant_id
    st.toast(body=f"Make assistant with id : {assistant_id}")
    thread_id = investor_assistant.make_thread_id(client)
    st.session_state["thread_id"] = thread_id


def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    if role == "user":
        role = "human"
    elif role == "assistant":
        role = "ai"
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


def get_answer(user_input):
    while st.session_state["run_status"] in [
        "start",
        "in_progress",
        "requires_action",
        "completed",
    ]:
        run = investor_assistant.get_run(
            client=st.session_state["client"],
            run_id=st.session_state["run_id"],
            thread_id=st.session_state["thread_id"],
        )
        st.session_state["run_status"] = run.status
        if st.session_state["run_status"] == "requires_action":
            investor_assistant.submit_tools_outputs(
                client=st.session_state["client"],
                run_id=st.session_state["run_id"],
                thread_id=st.session_state["thread_id"],
            )
        time.sleep(1)

        # completed
        if st.session_state["run_status"] == "completed":
            messages = investor_assistant.get_messages(
                client=st.session_state["client"],
                thread_id=st.session_state["thread_id"],
            )
            answer = messages[-1]["content"]
            save_message(answer, role="ai")
            send_message(
                answer,
                "ai",
                save=False,
            )
            st.download_button(
                "Download Answer",
                answer,
                f"{user_input}.txt",
            )
            break


st.set_page_config(
    page_title="Assistant GPT",
    page_icon="😾",
)

st.title("Assistant GPT")

with st.sidebar:
    api_key = None
    api_key = st.text_input("Put Your OpenAI API KEY")
    if api_key:
        if "assistant_id" in st.session_state:
            st.info(
                f"Your Assistant is ready ( id : {st.session_state['assistant_id']} )"
            )
        else:
            st.button("Make My AssistantAPI", on_click=make_assistant)

    st.info("This page operates based on the below Github repository.")
    st.link_button(
        label="Github", url="https://github.com/pjh5474/python_gpt/tree/streamlit"
    )


if "thread_id" in st.session_state:
    send_message(
        f"I'm ready! Ask about anything you want",
        "ai",
        save=False,
    )
    paint_history()
    user_input = st.chat_input("Ask a question")
    if user_input:
        send_message(user_input, "human")
        investor_assistant.send_message(
            client=st.session_state["client"],
            thread_id=st.session_state["thread_id"],
            content=user_input,
        )
        run_id = investor_assistant.make_run_id(
            client=st.session_state["client"],
            thread_id=st.session_state["thread_id"],
            assistant_id=st.session_state["assistant_id"],
        )
        st.session_state["run_id"] = run_id
        run = investor_assistant.get_run(
            client=st.session_state["client"],
            run_id=st.session_state["run_id"],
            thread_id=st.session_state["thread_id"],
        )
        st.session_state["run_status"] = run.status
        get_answer(user_input)


else:
    st.markdown(
        """
        Make and Use your own OpenAI Assistant API !
        Put your OpenAI API Key on sidebar, and ask AI what your are interested in.
        Your assistant will search keyword from duckduckgo & wikipedia.
    """
    )
    st.session_state["messages"] = []
