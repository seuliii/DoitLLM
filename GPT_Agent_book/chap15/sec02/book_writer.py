from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List

from utils import save_state, get_outline, save_outline
from datetime import datetime
from models import Task
from tools import retrieve, web_search, add_web_pages_hson_to_chroma

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
    task_history: List[Task]
    references:dict
    user_request : str

#현재 사용자의 요구사항을 파악하는 에이전트
def business_analyst(state: State):
    print("\n\n=========BUSINESS ANALYST=========")

    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 비즈니스 애널리스트로서,
        AI팀의 진행상황과 "사용자 요구 사항"을 토대로,
        현 시점에서 '지난 요구 사항(previous_user_request)'과 최근 사용자의 발언을 바탕으로 요구사항이 무엇인지 판단한다.
        지난 요구 사항은 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야하는지 결정한다.

        다음과 같은 템플릿 형태로 반환한다.
        '''
        -목표 : 0000 \n 방법 : 0000
        '''

        -------------------------------------
        *지난 요구 사항(previous_user_request)* : {previous_user_request}
        -------------------------------------
        사용자 최근 발언 : {user_last_comment}
        -------------------------------------
        참고 자료 : {references}
        -------------------------------------
        목차(outline) : {outline}
        -------------------------------------
        "messages" : {messages}
        """
    )
    
    ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

    messages = state["messages"]
    user_last_comment = None
    for m in messages[::-1]:
        if isinstance(m,HumanMessage):
            user_last_comment = m.content
            break
    
    inputs = {
        "previous_user_request" : state.get("user_request",None),
        "references" : state.get("references",{"queries" : [], "docs" :[]}),
        "outline" : get_outline(current_path),
        "messages" : messages,
        "user_last_comment" : user_last_comment

    }

    user_request = ba_chain.invoke(inputs)
    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message)
    messages.append(business_analyst_message)

    save_state(current_path,state)
    
    return{
        "messages" : messages,
        "user_request" : user_request
    }



def supervisor(state : State):
    print("\n\n=========SUPERVISOR=========")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI팀의 supervisor로서 AI팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고,
        사용자의 요구에 달성하기 위해 현재 해야 할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.
        -content_strategist : 사용자의 요구 사항이 명확해졌을 때 사용한다. AI팀의 콘텐ㅌ츠 전략을 결정하고,
        전체 책의 목차(outline)를 작성한다.
        -communicator: AI팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다.
        사용자에게 진행 상황을 보고하고, 다음 지시를 물어본다.
        -web_search_agent: 웹 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
        -vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.


        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.

        ---------------------------------------
        previous_outline : {outline}
        ---------------------------------------
        messages:
        {messages}
        """
    )

    supervisor_chain = supervisor_system_prompt | llm.with_structured_output(Task)

    messages = state.get("messages",[])

    inputs = {
        "messages" : messages,
        "outline" : get_outline(current_path)
    }

    task = supervisor_chain.invoke(inputs)
    task_history = state.get("task_history",[])
    task_history.append(task)

    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    return{
        "messages" : messages,
        "task_history" : task_history
    }

def supervisor_router(state: State):
    task = state['task_history'][-1]
    return task.agent

def vector_search_agent(state: State):
    print("\n\n=========VECTOR SEARCH AGENT=========")

    tasks = state.get("task_history",[])
    task = tasks[-1]
    if task.agent != "vector_search_agent":
        raise ValueError(f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n{task}")
    
    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent들이 수행한 작업을 바탕으로 목차(outline) 작성에
        필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.

        현재 목차(outline)를 작성하는 데 필요한 정보를 확보하기 위해,
        다음 내용을 활용해 적절한 벡터 검색을 수행하라.

        -검색 목적 : {mission}
        ----------------------------------
        -과거 검색 내용 : {references}
        ----------------------------------
        -이전 대화 내용: {messages}
        ----------------------------------
        -목차(outline) :{outline}  
        """
    )

    mission = task.description
    references = state.get("references",{"queries":[], "docs":[]})
    messages = state["messages"]
    outline = get_outline(current_path)

    inputs ={
        "mission" : mission,
        "references": references,
        "messages" : messages,
        "outline" : outline
    }
    llm_with_retriever = llm.bind_tools([retrieve])
    vector_search_chain = vector_search_system_prompt | llm_with_retriever

    search_plans = vector_search_chain.invoke(inputs)

    for tool_call in search_plans.tool_calls:
        print('--------------------',tool_call)

        args = tool_call["args"]
        query = args["query"]
        retrieved_docs = retrieve.invoke(args)

        references["queries"].append(query)
        references["docs"]+=retrieved_docs

    unique_docs = []
    unique_page_contents = set()

    for doc in references["docs"]:
        if doc.page_content not in unique_page_contents:
            unique_docs.append(doc)
            unique_page_contents.add(doc.page_content)
    references["docs"] = unique_docs

    print('-----------Queries')
    queries = references['queries']
    for query in queries:
        print(query)

    print('-----------References')
    for doc in references["docs"]:
        print(doc.page_content[:100])
        print('------------------')
    
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_task = Task(
        agent = "communicator",
        done = False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at = ""

    )
    tasks.append(new_task)

    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료 : {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)

    return {
        "messages" : messages,
        "task_history" : tasks,
        "references" : references
    }

def content_strategist(state: State):
    print("\n\n=========CONTENT STRATEGIST=========")
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
             너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist) fhtj,
             이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고,
             AI팀이 쓸 책의 세부 목차를 결정한다.

             지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
             목차를 작성하는데 필요한 정보는 "참고 자료"에 있으므로 활용한다.

             -----------------------
             -지난 목차 : {outline}
             -----------------------
             -이전 대화 내용 : {messages}
             -----------------------
             -참고 자료 : {references}

        """
    )
    
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state['messages']
    outline = get_outline(current_path)

    inputs = {
        "messages" : messages,
        "outline" : outline,
        "references" : state.get("references",{"queries":[],"docs":[]})
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

    task_history = state.get("task_history",[])

    if task_history[-1].agent != "content_strategist":
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_task = Task(
        agent = "communicator",
        done=False,
        description="AI팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다",
        done_at = ""
    )
    task_history.append(new_task)

    return {"messages" : messages, "task_history" : task_history}

#웹 검색 후 웹페이지 문서를 벡터디비에 자동으로 저장하고 벡터 검색
def web_search_agent(state: State):
    print("\n\n=========WEB SEARCH AGENT =========")

    tasks = state.get("task_history",[])
    task = tasks[-1]

    if task.agent != "web_search_agent":
        raise ValueError(f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다. \n {task}")
    
    web_search_system_prompt = PromptTemplate.from_template(
    """
    너는 다른 AI Agent들이 수행한 작업을 바탕으로,
    목차(outline) 작성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.

    현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 검색하라.
    
    -검색 목적 : {mission}
    ----------------------------------
    -과거 검색 내용 : {references}
    ----------------------------------
    -이전 대화 내용: {messages}
    ----------------------------------
    -목차(outline) :{outline}  
    ----------------------------------
    -현재 시각 : {current_time} 
    """
    )
    messages = state.get("messages",[])

    inputs = {
        "mission" : task.description,
        "references" : state.get("references",{"queries" : [],"docs":[]}),
        "messages" : messages,
        "outline" : get_outline(current_path),
        "current_time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    llm_with_web_search = llm.bind_tools([web_search])
    web_search_chain = web_search_system_prompt | llm_with_web_search
    search_plans = web_search_chain.invoke(inputs)

    queries = []

    for tool_call in search_plans.tool_calls:
        print('------------------',tool_call)
        args = tool_call["args"]

        queries.append(args["query"])
        _, json_path = web_search.invoke(args)
        print('json_path : ',json_path)

        add_web_pages_hson_to_chroma(json_path)
    
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    task_desc += f"다음 항목이 새로 추가되었다\n: {queries}"

    new_task= Task(
        agent="vector_search_agent",
        done=False,
        description=task_desc,
        done_at=""
    )

    tasks.append(new_task)

    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료 : {queries}"
    messages.append(AIMessage(msg_str))

    return{
        "messages":messages,
        "task_history":tasks
    }

#사용자와 대화하는 에이전트 communicator
def communicator(state : State):
    print("\n\n=========COMMUNICATOR=========")
    
    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의커뮤니케이터로서,
        AI팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.
        
        사용자도 outline(목차)를 이미 보고 있으므로, 다시 출력할 필요는 없다.

        outline : {outline}
        -----------------------------------------
        messages : {messages}
        """
    )

    system_chain = communicator_system_prompt | llm

    messages = state['messages']
    inputs = {"messages" : messages, "outline" : get_outline(current_path)}

    gathered = None

    print("\nAI\t: ", end="")
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end="")

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    
    messages.append(gathered)

    task_history = state.get("task_history",[])

    if task_history[-1].agent != "communicator":
        raise ValueError(f"communicator가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {"messages" : messages, "task_history": task_history}

#start -> communicator -> end
graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)
graph_builder.add_node("business_analyst", business_analyst)

graph_builder.add_edge(START, "business_analyst")
graph_builder.add_edge("business_analyst", "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist" : "content_strategist",
        "communicator" : "communicator",
        "vector_search_agent" : "vector_search_agent",
        "web_search_agent" : "web_search_agent"
    }
)
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "communicator")
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

                현재 시각
                은 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}이다.
            """
        )
    ],
    task_history=[],
    references={"queries":[],"docs":[]},
    user_request=""
)

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit','q','quit']:
        print("Bye")
        break

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n-------------------MESSAGE COUNT\t ', len(state["messages"]))
    references = state.get("references",{"queries":[], "docs":[]})
    save_state(current_path,state)

#HYBE와 JYP 비교하는 책 써줘. CEO와 경영전략에 대해 써줘