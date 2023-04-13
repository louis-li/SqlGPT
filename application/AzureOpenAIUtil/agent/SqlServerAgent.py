"""SQL agent."""
from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.sql_database import SQLDatabase
from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Extra, Field, validator
import os
from langchain.chains.llm import LLMChain
from langchain.llms.openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from typing import Any, List, Optional

from langchain.agents.agent_toolkits.base import BaseToolkit

SQL_PREFIX = """You are an agent designed to interact with a Microsoft Azure SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results using SELECT TOP in SQL Server syntax.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" as the answer.
        """

SQL_SUFFIX = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}"""



QUERY_CHECKER = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query."""



def ch(text: str) -> str:
    s = text if isinstance(text, str) else str(text)
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("\r", "").replace("\n", "<br>")

class HtmlCallbackHandler (BaseCallbackHandler):
    html: str = ""

    def get_and_reset_log(self) -> str:
        result = self.html
        self.html = ""
        return result
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        self.html += f"LLM prompts:<br>" + "<br>".join(ch(prompts)) + "<br>";

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        self.html += f"LLM prompts end:<br>" + "<br>".join(response) + "<br>";
        # pass

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.html += f"<span style='color:red'>LLM error: {ch(error)}</span><br>"

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        self.html += f"Entering chain: {ch(class_name)}<br>"

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        self.html += f"Finished chain<br>"

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        self.html += f"<span style='color:red'>Chain error: {ch(error)}</span><br>"

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        action: AgentAction,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        self.html += f"<span style='color:{color}'>{ch(action)}</span><br>"

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        self.html += f"{ch(observation_prefix)}<br><span style='color:{color}'>{ch(output)}</span><br>{ch(llm_prefix)}<br>"

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self.html += f"<span style='color:red'>Tool error: {ch(error)}</span><br>"

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        """Run when agent ends."""
        self.html += f"<span style='color:{color}'>{ch(text)}</span><br>"

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        self.html += f"<span style='color:{color}'>{ch(finish.log)}</span><br>"

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        # print(action.log)
        self.html += f"<span style='color:{color}'>{ch(action.log)}</span><br>"
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    # Override BaseTool.Config to appease mypy
    # See https://github.com/pydantic/pydantic/issues/4173
    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""

    name = "query_sql_db"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned. 
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "schema_sql_db"
    description = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling list_tables_sql_db first!
    
    Example Input: "table1, table2, table3"
    """

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(table_names.split(", "))

    async def _arun(self, table_name: str) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting tables names."""

    name = "list_tables_sql_db"
    description = "Input is an empty string, output is a comma separated list of tables in the database."

    def _run(self, tool_input: str = "") -> str:
        """Get the schema for a specific table."""
        return ", ".join(self.db.get_table_names())

    async def _arun(self, tool_input: str = "") -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class QueryCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm_chain: LLMChain = Field(
        default_factory=lambda: LLMChain(
            llm=AzureOpenAI(temperature=0, deployment_name=os.getenv('DEPLOYMENT_NAME')),
            prompt=PromptTemplate(
                template=QUERY_CHECKER, input_variables=["query", "dialect"]
            ),
        )
    )
    name = "query_checker_sql_db"
    description = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    """

    @validator("llm_chain")
    def validate_llm_chain_input_variables(cls, llm_chain: LLMChain) -> LLMChain:
        """Make sure the LLM chain has the correct input variables."""
        if llm_chain.prompt.input_variables != ["query", "dialect"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )
        return llm_chain

    def _run(self, query: str) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(query=query, dialect=self.db.dialect)

    async def _arun(self, query: str) -> str:
        return await self.llm_chain.apredict(query=query, dialect=self.db.dialect)



class SQLDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)
    callback_manager: Optional[BaseCallbackManager] = None,
    
    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QuerySQLDataBaseTool(db=self.db, callback_manager=self.callback_manager),
            InfoSQLDatabaseTool(db=self.db, callback_manager=self.callback_manager),
            ListSQLDatabaseTool(db=self.db, callback_manager=self.callback_manager),
            QueryCheckerTool(db=self.db, callback_manager=self.callback_manager),
        ]



def create_sql_agent(
    llm: BaseLLM,
    toolkit: SQLDatabaseToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = SQL_PREFIX,
    suffix: str = SQL_SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    early_stopping_method: str = "force",
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a sql agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions,
        input_variables=input_variables,
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        # verbose=verbose,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names,callback_manager=callback_manager, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=verbose,
        max_iterations=max_iterations,
        early_stopping_method=early_stopping_method,
        callback_manager=callback_manager
    )

