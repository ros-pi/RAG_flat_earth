import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore  # The pure-Python solution!
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- PACKAGE EXPLANATIONS ---
# 1. python-dotenv: Securely loads API keys from hidden files so they aren't pushed to GitHub.
from dotenv import load_dotenv

# 2. langchain_community: Contains integrations with third-party tools (like reading text files).
from langchain_community.document_loaders import TextLoader

# 3. langchain_text_splitters: Tools to chop text into manageable pieces for the AI's memory limit.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 4. langchain_google_genai: The official Google library to access Gemini's brain and embeddings.
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# 5. langchain_core: The core architecture. We use InMemoryVectorStore here because it is 
# written in 100% pure Python. We chose this over ChromaDB to prevent C++ SQLite crashes on Windows.
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the GOOGLE_API_KEY into the operating system's environment variables
load_dotenv()

# Helper function: Takes the raw chunks found by the database and glues them into one clean string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("--- Booting up the Flat Earth RAG Engine ---")

try:
    # STEP 1: INGESTION & CHUNKING
    print("1. Loading and Chunking data...")
    loader = TextLoader("data/flat_earth_lore.txt", encoding="utf-8")
    docs = loader.load()
    
    # WHY: LLMs have a "Context Window" limit. We cut the text into 300-character blocks.
    # The overlap=50 ensures that if a sentence is cut in half, the meaning isn't lost.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # STEP 2: EMBEDDING & VECTOR DATABASE
    print("2. Creating Embeddings and Database (Pure Python)...")
    # WHY: LLMs don't understand text; they understand math. The Embedding model turns our 
    # chunks into numerical coordinates (vectors) based on their semantic meaning.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # WHY: We load the vectors into RAM. When a user asks a question, this database will 
    # mathematically search for the 3 closest text chunks (k=3).
    vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # STEP 3: THE LLM & PROMPT (THE BRAIN & THE GUARDRAILS)
    print("3. Connecting to Gemini Chat...")
    # WHY temperature=0? We want the AI to be factual and deterministic, not creative.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # WHY this prompt? This is the core of RAG. We "handcuff" the AI to our data.
    # By forcing it to say "I don't know" if the data is missing, we eliminate hallucinations.
    prompt = PromptTemplate.from_template("""
    You are an expert on the Flat Earth theory. Answer based ONLY on the Context. 
    If not in Context, say "I don't know based on the texts."

    Context: {context}

    Question: {input}

    Answer:
    """)

    # STEP 4: LCEL ORCHESTRATION (The Modern LangChain Pipe)
    # WHY LCEL? It pipes the data cleanly: Database Search -> Prompt Format -> LLM -> String Output
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ System Ready!\n")

except Exception as e:
    print(f"\n❌ Setup Error: {e}")
    sys.exit(1)


# --- INTERACTIVE CHAT LOOP ---
if __name__ == "__main__":
    while True:
        try:
            user_question = input("🧑 You: ")
        except EOFError:
            break
            
        if user_question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        print("🤖 Agent: ", end="", flush=True)
        try:
            # WHY stream()? Instead of waiting 5 seconds for the whole block of text to generate,
            # this prints the answer word-by-word as Google generates it, creating a much better UI.
            for chunk in rag_chain.stream(user_question):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"\n❌ Chat Error: {e}")

# # Relevant Questions:
# # These test if the RAG is correctly retrieving the "private" data provided in the text.
# # What is the physical function of the 150-foot tall Ice Wall in Antarctica? 
# #   "Expected Answer: It surrounds the perimeter of the Earth to hold the oceans in.
# # According to this theory, why do objects fall to the ground if gravity is an illusion?
# #   Expected Answer: Because the flat Earth is constantly accelerating upwards at $9.8\text{ m/s}^2$.
# # How do the Sun and Moon create the seasons over the flat disc?
# #   Expected Answer: By shifting their orbits between the Tropics of Cancer and Capricorn at an altitude of 3,000 miles.
# # What is the 'Firmament' and what is its relationship to the stars?
# #   Expected Answer: It is an impenetrable dome that separates waters; the stars are localized points of light residing within it.
# # 
# # Irrelevant Questions
# # These test for "leakage." A good RAG should either say it doesn't know (based on the provided text) or explicitly state that the info isn't in the document.
# # What is the average surface temperature of Mars?
# #   Purpose: To see if the LLM pulls from its general training data instead of sticking to the provided "Flat Earth" context.
# # Who was the lead engineer for the Apollo 11 moon mission?
# #   Purpose: The text mentions the moon landing is "theatrical," but it doesn't name specific people. This tests if the model hallucinates a "lore-friendly" name or uses real-world facts.
# # How many calories are in a standard-sized gala apple?
# #   Purpose: A completely unrelated topic to ensure the RAG isn't trying to force the "lore" tone into everyday facts
# #
# # Other test_questions Examples:
# # What exactly is gravity according to the texts, and who guards the ice wall?
# # What is the shape of the earth? who guards the border? 
# # If a penguin and a lion fight, who wins?
# # What is the role of the moon and the sun in the flat earth model?
# # What is my name?