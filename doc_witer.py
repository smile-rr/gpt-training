import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from openai import OpenAI


client = OpenAI(base_url='https://api.xty.app/v1', api_key="sk-WHRCoUdd39GWq4RW084f16CeBaAc4f21BdD178F6Ba55Fd80")


prompt = """\
你是一个文档生成器，你能够根据输入的json数据结构生成文档，请把用户的输入转换成MarkDown格式文档。
文档应包括对数据整体的描述，以及每个字段的描述（使用表格），包括字段名、数据类型、是否必填、说明等信息。
"""


def generate_doc(text):
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


if __name__ == '__main__':
    # 打开文件读取json数据
    with open('data.json', 'r') as f:
        text = f.read()
        generate_doc(text)

