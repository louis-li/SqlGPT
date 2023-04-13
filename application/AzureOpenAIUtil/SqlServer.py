import urllib, os
from AzureOpenAIUtil.agent.SqlServerAgent import create_sql_agent, SQLDatabaseToolkit, SQLDatabase, HtmlCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.llms.openai import AzureOpenAI
from langchain.agents import AgentExecutor

class SqlServer:
    cb_handler = HtmlCallbackHandler()
    cb_manager = CallbackManager(handlers=[cb_handler])

    def __init__(self, llm, Server, Database, Username, Password, port=1433, odbc_ver=18, topK=10) -> None:
        
        odbc_conn = 'Driver={ODBC Driver '+ str(odbc_ver) + ' for SQL Server};Server=tcp:' + \
            Server + f',{port};Database={Database};Uid={Username};Pwd={Password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
        params = urllib.parse.quote_plus(odbc_conn)
        self.conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)


        db = SQLDatabase.from_uri(self.conn_str)
        self.toolkit = SQLDatabaseToolkit(db=db, callback_manager=self.cb_manager)
        # print(deploy_name)
        self.agent_executor = create_sql_agent(llm,
                toolkit=self.toolkit,
                verbose=True,
                topK = topK,
                callback_manager=self.cb_manager
            )
        
    def run(self, text: str):
        answer =  self.agent_executor.run(text)
        thought_process = self.cb_handler.get_and_reset_log()
        return answer, thought_process
        