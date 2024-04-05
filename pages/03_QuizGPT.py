import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

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
    retriever = WikipediaRetriever(top_k_results=2, lang=language)
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

Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

Each question should have 4 answers, three of them must be incorrect and on should be correct.

Use (o) to signal the correct answer.

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

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red | Yellow | Green | Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku | Tbilisi(o) | Manila | Beirut
         
    Question: When was Avatar released?
    Answers: 2007 | 2001 | 2009(o) | 1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o) | Painter | Actor | Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""",
        )
    ]
)

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
        language = st.selectbox(
            "Select Wikipedia Language",
            [
                "en",
                "ko",
            ],
        )
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = search_wikipedia(topic)

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
    )

questions_chain = {"context": format_docs} | questions_prompt | llm
formatting_chain = formatting_prompt | llm

final_chain = {"context": questions_chain} | formatting_chain | output_parser


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    return final_chain.invoke(_docs)


def make_quiz_form(response, score, is_submitted):
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                score += 1
                is_submitted = True
            elif value is not None:
                st.error("Wrong!")
                is_submitted = True

        st.form_submit_button()
        if is_submitted:
            st.write(f"Your score : {score} / {len(response['questions'])}")


def retry_quiz(response):
    retry = st.button("Retry Quiz")
    if retry:
        score = 0
        is_submitted = False
        make_quiz_form(response, score, is_submitted)


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
        response = run_quiz_chain(docs, topic if topic else file.name)
        score = 0
        is_submitted = False
        make_quiz_form(response, score, is_submitted)

        if is_submitted and score != len(response["questions"]):
            retry_quiz()

        elif is_submitted and score == len(response["questions"]):
            st.balloons()
