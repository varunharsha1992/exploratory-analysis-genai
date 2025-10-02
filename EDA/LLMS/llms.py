from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider, model_name, is_embedding=False):
    if provider == "openai":
        return ChatOpenAI(model=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "gemini":
        if is_embedding:
            return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))