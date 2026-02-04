from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List

from utils import save_state, get_outline, save_outline
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

#현재 폴더 경로 찾기
#랭그래프 이미지로 저장 및 추후 작업 결과 파일 저장 경로로 활용
filename = os.path.basename(__file__)    #현재 파일명 반환
absolute_path = os.path.abspath(__file__)   #현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) #현재 .py파일이 있는 폴더 경로

#모델
llm = ChatOpenAI(model = "gpt-4o")

class State(TypedDict):
    messages : List[AnyMessage | str]

def content_strategist(state: State):
    print("\n\n=========CONTENT STRATEGIST=========")
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
             너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist) fhtj,
             이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고,
             AI팀이 쓸 책의 세부 목차를 결정한다.

             지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

             -----------------------
             -지난 목차 : {outline}
             -----------------------
             -이전 대화 내용 : {messages}

        """
    )
    
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state['messages']
    outline = get_outline(current_path)

    inputs = {
        "messages" : messages,
        "outline" : outline
    }
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end ='')
    
    print()

    save_outline(current_path, gathered)

    content_strategist_message = f"[(Content Strategist)] 목차 작성 완료 :"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))


    return {"messages" : messages}


#사용자와 대화하는 에이전트 communicator
def communicator(state : State):
    print("\n\n=========COMMUNICATOR=========")
    
    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의커뮤니케이터로서,
        AI팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.
        
        사용자도 outline(목차)를 이미 보고 있으므로, 다시 출력할 필요는 없다.
        messages : {messages}
        """
    )

    system_chain = communicator_system_prompt | llm

    messages = state['messages']
    inputs = {"messages" : messages}

    gathered = None

    print("\nAI\t: ", end="")
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end="")

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    
    messages.append(gathered)

    return {"messages" : messages}

#start -> communicator -> end
graph_builder = StateGraph(State)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)

graph_builder.add_edge(START, "content_strategist")
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

#그래프 도식화
graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py','.png'))

state = State(
    messages = [
        SystemMessage(
            f"""
                너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가 팀이다.
                사용자가 사용하는 언어로 대화하라.

                현재 시각은 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}이다.
            """
        )
    ]
)

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit','q','quit']:
        print("Bye")
        break

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n-------------------MESSAGE COUNT\t ', len(state["messages"]))
    
    save_state(current_path,state)

#HYBE와 JSP의 경영 전략과 기업 문화에 대한 책을 써줘