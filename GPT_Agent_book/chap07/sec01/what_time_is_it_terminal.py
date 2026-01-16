from gpt_functions import get_current_time, tools
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def get_ai_response(messages,tools=None):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools= tools
    )
    return response

messages = [
    {"role" : "system", "content" : "너는 사용자를 도와주는 상담사야."}
]

while True:
    user_input = input("사용자\t : ")
    
    if user_input == "exit":
        break

    messages.append({"role" : "user", "content" : user_input})

    ai_response = get_ai_response(messages, tools = tools)
    ai_message = ai_response.choices[0].message

    tool_calls = ai_message.tool_calls
    if tool_calls:
        
        tool_name = tool_calls[0].function.name
        tool_call_id = tool_calls[0].id

        arguments = json.loads(tool_calls[0].function.arguments)

        if tool_name == "get_current_time":
            messages.append(
                {
                    "role" : "function",
                    "tool_call_id" : tool_call_id,
                    "name" : tool_name,
                    "content" : get_current_time(timezone=arguments['timezone']),
                }
            )

    ai_response = get_ai_response(messages,tools=tools)
    ai_message = ai_response.choices[0].message

    messages.append(ai_message)

    print("AI\t : " + ai_message.content)