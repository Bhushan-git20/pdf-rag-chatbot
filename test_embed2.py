import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    res = embeddings.embed_query("what is the pdf about")
    print("gemini-embedding-001 length:", len(res))
except Exception as e:
    print("gemini-embedding-001 ERROR:", e)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    res = embeddings.embed_query("what is the pdf about")
    print("embedding-001 length:", len(res))
except Exception as e:
    print("embedding-001 ERROR:", e)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    res = embeddings.embed_query("what is the pdf about")
    print("text-embedding-004 length:", len(res))
except Exception as e:
    print("text-embedding-004 ERROR:", e)
