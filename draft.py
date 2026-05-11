import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load your GOOGLE_API_KEY from the .env file
load_dotenv()

# 1. To use the Chat Model (The Brain)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. To use the Embeddings (For your Vector Database)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# Quick test
response = llm.invoke("Are you connected?")

print(response.content)