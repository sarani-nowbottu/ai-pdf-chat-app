from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_qa_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # FIXED: gemini-2.0-flash was shut down June 1, 2026
        temperature=0.3
    )
    return llm