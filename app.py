import os
import json
import chainlit as cl

from operator import itemgetter
from dotenv import load_dotenv

from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import FunctionMessage, HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import StrOutputParser

from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END

from searches import GoodReadsSearch
from utils import AgentState

async def call_model(state: AgentState, config: RunnableConfig):
  messages = state["messages"]
  response = await model.ainvoke(messages, config)
  return {"messages" : [response]}

def call_tool(state):
  last_message = state["messages"][-1]

  action = ToolInvocation(
      tool=last_message.additional_kwargs["function_call"]["name"],
      tool_input=json.loads(
          last_message.additional_kwargs["function_call"]["arguments"]
      )
  )

  response = tool_executor.invoke(action)

  function_message = FunctionMessage(content=str(response), name=action.tool)

  return {"messages" : [function_message]}

def should_continue(state):
  last_message = state["messages"][-1]

  if "function_call" not in last_message.additional_kwargs:
    return "end"

  return "continue"

load_dotenv()

REDDIT_CLIENT_ID = os.environ["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = os.environ["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT = os.environ["REDDIT_USER_AGENT"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

tool_belt = [
    DuckDuckGoSearchRun(),
    RedditSearchRun(
        api_wrapper=RedditSearchAPIWrapper(
            reddit_client_id=REDDIT_CLIENT_ID,
            reddit_client_secret=REDDIT_CLIENT_SECRET,
            reddit_user_agent=REDDIT_USER_AGENT,
        )
    ),
    GoodReadsSearch()
]

tool_executor = ToolExecutor(tool_belt)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
functions = [convert_to_openai_function(t) for t in tool_belt]
model = model.bind_functions(functions)

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue" : "action",
        "end" : END
    }
)
workflow.add_edge("action", "agent")

app = workflow.compile()

@cl.on_chat_start
async def start_chat():
    """
    """
    cl.user_session.set("agent", app)

@cl.on_message  
async def main(message: cl.Message):
    """
    """
    agent = cl.user_session.get("agent")
    inputs = {"messages" : [HumanMessage(content=str(message.content))]}
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    config = RunnableConfig(callbacks=[cb])

    msg = cl.Message(content="")   
    await msg.send()    

    async for event in agent.astream_events(inputs, config=config, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            await msg.stream_token(event["data"]["chunk"].content)

    await msg.update()