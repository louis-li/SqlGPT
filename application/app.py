import datetime
import json
import os
import traceback
import asyncio
import dotenv
import requests
from flask import Flask, request, render_template, send_from_directory, jsonify
from langchain.llms import AzureOpenAI

from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from werkzeug.utils import secure_filename
from AzureOpenAIUtil import SqlServer
import openai

from error import bad_request

# loading the .env file
dotenv.load_dotenv()

import platform

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# load the prompts
with open("prompts/combine_prompt.txt", "r") as f:
    template = f.read()

with open("prompts/combine_prompt_hist.txt", "r") as f:
    template_hist = f.read()

with open("prompts/question_prompt.txt", "r") as f:
    template_quest = f.read()

with open("prompts/chat_combine_prompt.txt", "r") as f:
    chat_combine_template = f.read()

with open("prompts/chat_reduce_prompt.txt", "r") as f:
    chat_reduce_template = f.read()

if os.getenv("API_KEY") is not None:
    api_key_set = True
else:
    api_key_set = False


app = Flask(__name__)
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION") #openai api version
llm=AzureOpenAI(temperature=0, deployment_name=os.getenv('DEPLOYMENT_NAME'))
sql_agent = SqlServer.SqlServer(llm, 
                                Server=os.getenv('AZURE_SQL_SERVER'), 
                                Database=os.getenv('AZURE_SQL_DATABASE'), 
                                Username=os.getenv('AZURE_SQL_USERNAME'), 
                                Password=os.getenv('AZURE_SQL_PASSWORD'), 
                                topK=15)

async def async_generate(chain, question, chat_history):
    result = await chain.arun({"question": question, "chat_history": chat_history})
    return result

def run_async_chain(chain, question, chat_history):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = {}
    try:
        answer = loop.run_until_complete(async_generate(chain, question, chat_history))
    finally:
        loop.close()
    result["answer"] = answer
    return result

@app.route("/")
def home():
    return render_template("index.html", api_key_set=api_key_set)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    question = data["question"]
    history = data["history"]
    print('-' * 5)


    # use try and except  to check for exception
    try:
        # check if the vectorstore is set
        
        # create a prompt template
        # if history:
        #     history = json.loads(history)
        #     template_temp = template_hist.replace("{historyquestion}", history[0]).replace("{historyanswer}",
        #                                                                                    history[1])
        #     c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=template_temp,
        #                               template_format="jinja2")
        # else:
        #     c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=template,
        #                               template_format="jinja2")

        # q_prompt = PromptTemplate(input_variables=["context", "question"], template=template_quest,
        #                           template_format="jinja2")
        # llm = AzureChatOpenAI(openai_api_key=api_key,
        #                         openai_api_base=api_base, 
        #                         openai_api_type=os.getenv("OPENAI_API_TYPE"),
        #                         openai_api_version=os.getenv("OPENAI_API_VERSION"),
        #                         deployment_name=os.getenv("DEPLOYMENT_NAME"))
        # messages_combine = [
        #     SystemMessagePromptTemplate.from_template(chat_combine_template),
        #     HumanMessagePromptTemplate.from_template("{question}")
        # ]
        # p_chat_combine = ChatPromptTemplate.from_messages(messages_combine)
        # messages_reduce = [
        #     SystemMessagePromptTemplate.from_template(chat_reduce_template),
        #     HumanMessagePromptTemplate.from_template("{question}")
        # ]
        # p_chat_reduce = ChatPromptTemplate.from_messages(messages_reduce)

        # question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        # doc_chain = load_qa_chain(llm, chain_type="map_reduce", combine_prompt=p_chat_combine, )
        # chat_history = []
        # result = chain({"question": question, "chat_history": chat_history})
        answer = sql_agent.run(question)
        print(answer)

        return {'answer':answer}
    except Exception as e:
        # print whole traceback
        traceback.print_exc()
        print(str(e))
        return bad_request(500, str(e))


@app.route("/api/docs_check", methods=["POST"])
def check_docs():
    raise NotImplementedError


@app.route('/api/combine', methods=['GET'])
def combined_json():
    return ''

@app.route('/api/upload', methods=['POST'])
def upload_file():
    return {"status": 'ok'}

# @app.route('/api/task_status', methods=['GET'])
def task_status():
    raise NotImplementedError

# ### Backgound task api
@app.route('/api/upload_index', methods=['POST'])
def upload_index_files():
    raise NotImplementedError


@app.route('/api/download', methods=['get'])
def download_file():
    raise NotImplementedError


@app.route('/api/delete_old', methods=['get'])
def delete_old():
    raise NotImplementedError


# handling CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5010)
