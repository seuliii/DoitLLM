from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(model="deepseek-r1:14b")

messages = [
    SystemMessage("너는 사용자의 질문에 한국어로 답변해야 한다.")
]

while True:
    user_input = input("사용자 : ")

    if user_input == ["exit","quit","q"]:
        print("Good bye")
        break
    
    messages.append(HumanMessage(user_input))
    
    response = llm.stream(messages)
    ai_message = None
    for chunk in response:
        print(chunk.content,end='')
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk
    print('')

    message_only = ai_message.content.split("</think>")[1].strip()
    messages.append(message_only)

