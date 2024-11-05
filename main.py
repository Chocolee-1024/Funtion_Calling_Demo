from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uvicorn

# python FastAPI 後端框架
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces"
)
# 設定 CORS(跨來源資源共享)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["*"]
)

async def generated_response(context: str):
    nomal_ollama = ChatOllama(
        model = "llama3.2",
        temperature = 0
    )



    translated = await nomal_ollama.invoke(f"Please help me translate the following sentence into English, and do not respond with any other text, including quotation marks.\n\n {context}")
    print(translated)

    # 建立 prompt 模板
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "您的系統提示"),
    #     ("human", "{incontextput}")
    # ])

    # 建立 chain
    # chain = prompt | model
    
    return output
    

# 使用者輸入 API
@app.post("/api/v1/demo")
async def main(payload: dict):
    print(payload.get("context"))
    context = payload.get("context")
    generate = await generated_response(context)
    return StreamingResponse(payload, media_type="application/json")

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)