import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama


def get_sunrise_sunset_times(city: str):
    """
    A function that returns the time of sunrise and sunset at the present moment, for a given city, in the form of a list: [sunrise_time, sunset_time].
    
    Args:
        city: The city to get the sunrise and sunset times for.
    """
    
    return ["6:00 AM", "6:00 PM"]

def get_flight_info_tool(DepartureAirportID: str, ArrivalAirportID: str, ScheduleStartDate: str, ScheduleEndDate: str) -> list:
    """
    Returns flight information based on the provided request.

    Args:
        ScheduleStartDate: The schedule start date in the format 'YYYY-MM-DD'.
        ScheduleEndDate: The schedule end date in the format 'YYYY-MM-DD'.
        DepartureAirportID: The departure airport code.
        ArrivalAirportID: The question should refer to the airport code for a single destination, not the code for all airports in a region.
    
    Returns:
        list: A list of flight information dictionaries matching the request criteria.
    """
    # 這裡應該是實際查詢航班資訊的代碼
    # 為了示例，我們返回一個固定值

    url = "https://davinci-airplane.nutc-imac.com/api/v1/airplane/flight"
    body = {
        "DepartureAirportID": DepartureAirportID,
        "ArrivalAirportID": ArrivalAirportID,
        "ScheduleStartDate": ScheduleStartDate,
        "ScheduleEndDate": ScheduleStartDate
    }
    # 發送 POST 請求
    response = requests.post(url, json=body)
    # 檢查回應狀態碼
    if response.status_code == 200 and response != "":
        # 解析 JSON 回應
        flight_info = response.json()
    else:
        flight_info = response.text
    return flight_info

def get_weather_info_tool(location: str, days: str) -> list:
    """
    Returns weather information based on the provided request.

    Args:
        location: The city to get the weather for.
        days: The number of days for the weather forecast (e.g. 1).

    Returns:
        list: A list of flight information dictionaries matching the request criteria.
    """
    import requests        
    url = "https://davinci-weather.nutc-imac.com/api/v1/weather"
    body = {
        "q": location,
        "days": days
    }

    # 發送 POST 請求
    response = requests.post(url, json=body)

    # 檢查回應狀態碼
    if response.status_code == 200:
        # print(response.text)  # 打印響應內容
        # 解析 JSON 回應
        # print("response"*10, response)
        result = response.json()
        # print("flight_info", weather_info)
    else:
        print(f"請求失敗，狀態碼: {response.status_code}")
        print(response.text)
        result = response.text
    return "result"

# # langchain bind_tools底層用到 convert_pydantic_to_openai_tool convert_pydantic_to_openai_function[https://github.com/langchain-ai/langchain/blob/0ebddabf7ddafab30fefcbfae2148bf8e0fca42f/libs/core/langchain_core/utils/function_calling.py#L83]


model = AutoModelForCausalLM.from_pretrained(
    "DiTy/gemma-2-9b-it-function-calling-GGUF",
    device_map="auto",
    torch_dtype=torch.bfloat16,  # use float16 or float32 if bfloat16 is not available to you.
    cache_dir="/home/ubuntu/bowei/huggingface_model_download/model_download/cache",  # optional
    # gguf_file="gemma-2-9B-it-function-calling-Q6_K.gguf"
)
tokenizer = AutoTokenizer.from_pretrained(
    "DiTy/gemma-2-9b-it-function-calling-GGUF",
    cache_dir="/home/ubuntu/bowei/huggingface_model_download/model_download/cache",  # optional
    # gguf_file="gemma-2-9B-it-function-calling-Q6_K.gguf"
)

# Case 1 今天台北天氣如何(O)
# Case 2 今天東京天氣如何(O)
# Case 3 今天天氣如何(X)
# Case 4 幫我查詢2024年10月28號台北到大阪的航班(O)
# Case 5 幫我查詢2024年10月28號到大阪的航班(X)
# Case 6 幫我查詢2024年10月28號台北起飛的航班(X)
# Case 7 幫我查詢台北到大阪的航班(X)
history_messages = [
    # {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required - "},
    {"role": "user", "content": "幫我查詢2024年10月28號台北起飛的航班"},
]

inputs = tokenizer.apply_chat_template(
    history_messages,
    tokenize=False,
    add_generation_prompt=True,  # adding prompt for generation
    tools=[get_sunrise_sunset_times, get_weather_info_tool, get_flight_info_tool],  # our functions (tools)
)

# print("inputs", inputs)


terminator_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<end_of_turn>"),
]

prompt_ids =  tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt').to(model.device)

generated_ids = model.generate(
    prompt_ids,
    max_new_tokens=512,
    eos_token_id=terminator_ids,
    bos_token_id=tokenizer.bos_token_id,
)
generated_response = tokenizer.decode(generated_ids[0][prompt_ids.shape[-1]:], skip_special_tokens=False)  # skip_special_tokens=False for debug

print(generated_response)




















# PATH_TO_MODEL_DIR = "/home/ubuntu/bowei/huggingface_model_download/model_download/cache"
# # 用transformer 先定義預訓練完的，不知道piepeline在幹嘛
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device_map="auto",
#     temperature=0,
#     max_new_tokens=512
# )

# # 用langchain的hugging face pipeline轉成langchain可以用的樣子[https://python.langchain.com/api_reference/community/llms/langchain_community.llms.huggingface_pipeline.HuggingFacePipeline.html]
# hugging_face_pipeline = HuggingFacePipeline(pipeline=pipe)

# 定義一個prompt模板。
# template = """You are a helpful assistant with access to the following functions. Use them if required
# {question} 
# """

# # 定義問題和上下文。
# question_p = """Hi, can you tell me the time of sunrise in Los Angeles?"""
# # context_p = """ On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore."""

# # 使用模板和定義的問題與上下文創建一個LLMChain實例。
# prompt = PromptTemplate(template=template, input_variables=["question", "context"], tools=[get_weather, get_sunrise_sunset_times])
# print("prompt", prompt)
# llm_chain = prompt | hugging_face_pipeline

# # 執行LLMChain，得到回應。
# response = llm_chain.invoke({"question":question_p})

# print(response)

# # @tool
# # def get_temperature_humidity_tool(time: str) -> str:
# #     """Returns time string in the format.

# #     Args:
# #         time (str): The input time string in the format 'YYYY-MM-DD HH:MM:SS'.

# #     Returns:
# #         str: The same time string in the format 'YYYY-MM-DD HH:MM:SS'.
# #     """
# #     return "OKOK"

# # @tool
# # def get_weather_info_tool(location: str, days: int) -> list:
# #     """Returns weather information based on the provided request.

# #     Args:
# #         location: This parameter is a string representing a geographical location. (e.g., "Taoyuan"、"Taichung").
# #         days: The number of days for the weather forecast (e.g., 1).

# #     """
# #     return "OK"

# # def get_weather(city: str):
# #     """
# #     A function that returns the weather in a given city.
    
# #     Args:
# #         city: The city to get the weather for.
# #     """
# #     import random
    
# #     return "sunny" if random.random() > 0.5 else "rainy"


# # def get_sunrise_sunset_times(city: str):
# #     """
# #     A function that returns the time of sunrise and sunset at the present moment, for a given city, in the form of a list: [sunrise_time, sunset_time].
    
# #     Args:
# #         city: The city to get the sunrise and sunset times for.
# #     """
    
# #     return ["6:00 AM", "6:00 PM"]

# # question = "請問台北天氣如何？"

# # chat = [
# #         {"role": "user", "content": "question: 請問台北天氣如何？"}
# # ]

# # ollama = ChatOllama(model="imac/gemma-2-9b-it-function-calling:q6_k")

# # tools = [
# #     Tool(
# #         name="get_temperature_humidity",
# #         func=get_temperature_humidity_tool,
# #         description="Get temperature and humidity information"
# #     ),
# #     Tool(
# #         name="get_weather_info",
# #         func=get_weather_info_tool,
# #         description="Get weather information for a specific location and number of days"
# #     )
# # ]

# # print(ollama.invoke(chat, tools=tools))


# 用transformer 先定義預訓練完的，不知道piepeline在幹嘛
# PATH_TO_MODEL_DIR = "/home/ubuntu/bowei/huggingface_model_download/model_download/cache"

# generation_pipeline = pipeline(
#     "text-generation",
#     model="DiTy/gemma-2-9b-it-function-calling-GGUF",
#     model_kwargs={
#         "torch_dtype": torch.bfloat16,  # use float16 or float32 if bfloat16 is not supported for you. 
#         "cache_dir": PATH_TO_MODEL_DIR,  # OPTIONAL
#     },
#     device_map="auto",
# )

# history_messages = [
#     {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required - "},
#     {"role": "user", "content": "你可以告訴我台北什麼時候日落嗎?"},
#     {"role": "function-call", "content": '{"name": "get_sunrise_sunset_times", "arguments": {"city": "Los Angeles"}}'},
#     {"role": "function-response", "content": '{"times_list": ["6:00 AM", "6:00 PM"]}'},
# ]

# inputs = generation_pipeline.tokenizer.apply_chat_template(
#     history_messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     tools=[get_weather, get_sunrise_sunset_times],
# )
# print("inputs", inputs)
# terminator_ids = [
#     generation_pipeline.tokenizer.eos_token_id,
#     generation_pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
# ]
# print("terminator_ids", terminator_ids)
# outputs = generation_pipeline(
#     inputs,
#     max_new_tokens=512,
#     eos_token_id=terminator_ids,
# )

# print(outputs[0]["generated_text"][len(inputs):])