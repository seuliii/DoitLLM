from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3.2:3b")

messages = [
    SystemMessage("You are a helpful assistant. ")
]

while True:
    user_input = input("사용자 : ")
    
    if user_input == 'exit':
        break
    messages.append(
        HumanMessage(user_input)
    )
    ai_response = llm.invoke(messages)

    messages.append(ai_response)

    print("AI : " + ai_response.content)
