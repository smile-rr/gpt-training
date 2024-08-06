import asyncio
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


os.environ["OPENAI_API_KEY"] = "sk-WHRCoUdd39GWq4RW084f16CeBaAc4f21BdD178F6Ba55Fd80"
os.environ["OPENAI_API_BASE"] = "https://api.xty.app/v1"


async def generate_doc_lc(doc):
    prompt_template = """你是一个文档生成器，你能够根据输入的json数据结构生成文档，请把用户的输入转换成MarkDown格式文档。
文档应包括对数据整体的描述，以及每个字段的描述（使用表格），包括字段名、数据类型、是否必填、说明等信息。

{input}
"""
    pmt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
    parser = StrOutputParser()

    chain = pmt | llm | parser

    async for chunk in chain.astream({"input": doc}):
        print(chunk, flush=True, end='')
    print('\n')


if __name__ == '__main__':
    # 打开文件读取json数据
    with open('data.json', 'r') as f:
        text = f.read()
        asyncio.run(generate_doc_lc(text))

