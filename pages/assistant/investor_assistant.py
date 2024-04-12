from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.retrievers import WikipediaRetriever
import json
from typing import Type
from openai import OpenAI


def get_duckduckgo_data(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    query = inputs["query"]
    return ddg.run(query)


def get_wikipedia_data(inputs):
    retriever = WikipediaRetriever(top_k_results=2)
    query = inputs["query"]
    docs = retriever.get_relevant_documents(query)
    docs = "\n\n".join(document.page_content for document in docs)
    return docs


functions_map = {
    "get_duckduckgo_data": get_duckduckgo_data,
    "get_wikipedia_data": get_wikipedia_data,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "get_duckduckgo_data",
            "description": "From given query, find a main keyword and get data from duckduckgo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Searching query which contains a main keyword.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wikipedia_data",
            "description": "From given query, find a main keyword and get data from wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Searching query which contains a main keyword.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def make_assistant_id(client: Type[OpenAI]):
    assistant = client.beta.assistants.create(
        name="Investor Assistant",
        instructions="""You are a very meticulous Data searcher.

                You want to find information from every source possible.

                You can find information from both DuckDuckGo and Wikipedia.

                After searching data from duckduckgo and wikipedia, You should write a professional report about main keyword.""",
        model="gpt-3.5-turbo",
        tools=functions,
    )
    return assistant.id


def make_thread_id(client: Type[OpenAI]):
    thread = client.beta.threads.create()
    return thread.id


def make_run_id(client: Type[OpenAI], thread_id, assistant_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return run.id


def get_run(client: Type[OpenAI], run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(client: Type[OpenAI], thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def get_messages(client: Type[OpenAI], thread_id):
    thread_messages = client.beta.threads.messages.list(thread_id=thread_id).data
    thread_messages.reverse()
    messages_list = [
        {"role": thread_message.role, "content": thread_message.content[0].text.value}
        for thread_message in thread_messages
    ]
    return messages_list


def get_tools_outputs(client: Type[OpenAI], run_id, thread_id):
    run = get_run(client, run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        outputs.append(
            {
                "tool_call_id": action_id,
                "output": functions_map[function.name](json.loads(function.arguments)),
            }
        )
    return outputs


def submit_tools_outputs(client: Type[OpenAI], run_id, thread_id):
    outputs = get_tools_outputs(client, run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )
