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

api_key_set = True


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

        answer, thought = sql_agent.run(question)
        print(thought)

        return {'answer':answer, 'thought': thought}
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
