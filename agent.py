import os

from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import ConnectionParams, IndexParams
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI


DB_KEY = 'eSWPfALwd1DXTMyoZFR4B8piVAWj5kviWnbgHwk7'
DB_HOST = 'http://lb-defu84qr-0n2e7p4s8fatm9bh.clb.ap-beijing.tencentclb.com:40000'

os.environ["OPENAI_API_KEY"] = "sk-WHRCoUdd39GWq4RW084f16CeBaAc4f21BdD178F6Ba55Fd80"
os.environ["OPENAI_API_BASE"] = "https://api.xty.app/v1"


vector_db = TencentVectorDB(
    None,
    connection_params=ConnectionParams(
        url=DB_HOST,
        key=DB_KEY,
    ),
    index_params=IndexParams(768, replicas=0),
    t_vdb_embedding='bge-base-zh',
    database_name='online_shop',
    collection_name='product',
    collection_description='product descriptions for an online shop'
)

retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

ORDER_ITEMS = []


def search_product(text: str) -> str:
    """根据text参数查询较为匹配的产品详细信息，信息中包含产品ID，名称，简介，价格等等。查询结果可能为多个产品信息，它们之间用换行符分隔。"""
    docs = retriever.invoke(text)
    return "\n\n".join(doc.page_content for doc in docs)


def place_order(product_id, sku_id, quantity, price) -> dict:
    """根据product_id, sku_id, quantity参数生成订单，返回订单详情。"""
    global ORDER_ITEMS
    item = {
        "product_id": product_id,
        "sku_id": sku_id,
        "quantity": quantity,
        "price": price,
    }
    ORDER_ITEMS.append(item)
    return item


def check_out() -> str:
    """结账，返回订单总金额。"""
    total_price = 0
    for order_item in ORDER_ITEMS:
        quantity = order_item["quantity"]
        price = order_item["price"]
        total_price += float(price) * quantity
    return f"订单总金额: {total_price}"


# Prompts

SYSTEM_PROMPT = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
"""

HUMAN_PROMPT = """{input}

{agent_scratchpad}
 (reminder to respond in a JSON blob no matter what)"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
])

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)

tools = [
    StructuredTool.from_function(search_product, name="search_product", description="Search for products based on a query"),
    StructuredTool.from_function(place_order, name="place_order", description="Place an order for a product"),
    StructuredTool.from_function(check_out, name="check_out", description="Check out the order")
]

agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)

# result = agent_executor.invoke({"input": "有什么婴儿玩具可以推荐？"})

result = agent_executor.invoke({"input": "我要买一套白色的婴儿防护围栏。"})
print(result['output'])


