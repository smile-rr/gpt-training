from openai import OpenAI

client = OpenAI(base_url='https://api.xty.app/v1', api_key="sk-WHRCoUdd39GWq4RW084f16CeBaAc4f21BdD178F6Ba55Fd80")


def translate(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个翻译官，你能够根据输入的文本完成中英文之间的翻译，请把用户的输入翻译成另一种语言。"},
            {"role": "user", "content": text}
        ],
        stream=False,
    )
    print(response.choices[0].message.content)


if __name__ == '__main__':
    while True:
        text = input('请输入要翻译的文本：')
        if text == 'exit':
            break
        translate(text)

