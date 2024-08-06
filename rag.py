import os
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import ConnectionParams, IndexParams

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
    drop_old=False,
    collection_description='product descriptions for an online shop'
)

retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def prepare():
    documents = TextLoader('dataset/product.txt').load_and_split(CharacterTextSplitter(
        separator="\n\n",
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    ))
    vector_db.add_documents(documents)


def retrieve(text):
    relevant_docs = retriever.invoke(text)
    return relevant_docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# if __name__ == '__main__':
#     print(format_docs(retrieve("有什么玩具适合一岁宝宝玩")))


def augment_generation(text):
    template = """
        使用以下上下文片段回答用户问题，如果你不知道答案，只需说你不知道，不要试图编造答案。

        {context}

        问题：{question}
        答案：
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.0, model="gpt-4")
    context = format_docs(retriever.invoke(text))
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"context": context, "question": text})
    print(result['text'])


def augment_generation_pineline(text):
    template = """
        使用以下上下文片段回答用户问题，如果你不知道答案，只需说你不知道，不要试图编造答案。
        
        {context}
        
        问题：{question}
        答案：
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.0, model="gpt-4")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    for chunk in rag_chain.stream(text):
        print(chunk, end="", flush=True)


def run():
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        augment_generation_pineline(question)
        print("\n")


if __name__ == '__main__':
    # prepare()
    run()
