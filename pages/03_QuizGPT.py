import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

quiz_function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz. (o) signal is used to show the correct answer. Delete (o) signal when making questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


st.set_page_config(
    page_title="Quiz GPT",
    page_icon="🧐",
)

st.title("Quiz GPT")


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=150,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wikipedia(topic):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(topic)
    return docs


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that is role playing as a teacher.

Based ONLY on the following context make minimum 3 to maximum 10 'DIFFERENT' questions to test the user's knowledge about the text. 

Each question should have 4 answers, three of them must be incorrect and on should be correct.

Use (o) to signal the correct answer.

There are three levels of difficulty for the quiz: easy, normal difficult.
When you make a quiz, you have to make it at this difficulty level : {difficulty}.

Question examples:

Question: What is the color of the ocean?
Answers: Red | Yellow | Green | Blue(o)

Question: What is the capital of Georgia?
Answers: Baku | Tbilisi(o) | Manila | Beirut

Question: When was Avatar released?
Answers: 2007 | 2001 | 2009(o) | 1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(o) | Painter | Actor | Model

Your turn!

context: {context}
""",
        ),
    ]
)

difficulty = "easy"

with st.sidebar:
    api_key = None
    api_key = st.text_input("Put Your OpenAI API KEY")
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["docx", "txt", "pdf"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = search_wikipedia(topic)

    difficulty = st.selectbox(
        "Select Quiz Difficulty",
        [
            "easy",
            "hard",
        ],
    )

    st.info("This page operates based on the below Github repository.")
    st.link_button(
        label="Github", url="https://github.com/pjh5474/python_gpt/tree/streamlit"
    )

if api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        api_key=api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            quiz_function,
        ],
    )

    quiz_chain = questions_prompt | llm


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    docs = format_docs(_docs)
    response = quiz_chain.invoke(
        {
            "context": docs,
            "difficulty": difficulty,
        }
    )
    response = response.additional_kwargs["function_call"]["arguments"]

    return json.loads(response)


score = 0
is_submitted = False
st.session_state["clear"] = None

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
    
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    if not api_key:
        st.info("Please input your OpenAI KEY")
    else:
        response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
        quiz_form = st.form(
            "questions_form",
            clear_on_submit=True if st.session_state["clear"] == True else False,
        )
        for question in response["questions"]:
            quiz_form.write(question["question"])
            value = quiz_form.radio(
                "Select a correct answer.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                quiz_form.success("Correct!")
                score += 1
                is_submitted = True
            elif value is not None:
                quiz_form.error("Wrong!")
                is_submitted = True

        if is_submitted and score != len(response["questions"]):
            st.session_state["clear"] = True
            quiz_form.form_submit_button("Retry")
        else:
            quiz_form.form_submit_button("Submit")
        if is_submitted:
            quiz_form.write(f"Your score : {score} / {len(response['questions'])}")

        if is_submitted and score == len(response["questions"]):
            st.session_state["clear"] = False
            quiz_form.balloons()
