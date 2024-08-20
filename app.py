import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import openai

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Document Loading and Splitting
def load_and_split_documents(urls):
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        all_splits += text_splitter.split_documents(data)
    return all_splits

# Set up embeddings and vector store
def setup_vector_store(documents):
    local_embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(documents=documents, embedding=local_embeddings)

# Initialize OpenAI Chat model
model = ChatOpenAI(model="gpt-4")

# Create RAG chain
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: "\n\n".join(doc.page_content for doc in input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

# Load and split documents
urls = ["https://www.meraevents.com/faq", "https://www.meraevents.com/pricing"]
documents = load_and_split_documents(urls)
vectorstore = setup_vector_store(documents)

# Create Flask app
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Perform similarity search and get response
    docs = vectorstore.similarity_search(question)
    response = chain.invoke({"context": docs, "question": question})
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)
