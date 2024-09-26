# Warning control
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import dict_to_elements
import os
from io import StringIO 
from lxml import etree
import pandas as pd

# Load environment variables
load_dotenv()

# API keys
Unstructured_API_KEY = "0tDZmwk5YLWXWmwm4y09QMhfdfFagg"
Unstructured_API_URL = "https://api.unstructuredapp.io/general/v0/general"

# Initialize the Unstructured Client
s = UnstructuredClient(
    api_key_auth=Unstructured_API_KEY,
    server_url=Unstructured_API_URL,
)

### Preprocess the PDF
filename = r"data\regdoclmEQ4.pdf"

with open(filename, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(
    files=files,
    strategy="hi_res",
    hi_res_model_name="yolox",
    pdf_infer_table_structure=True,
    skip_infer_table_types=[],
)

try:
    resp = s.general.partition(req)
    
    # Check if the response contains any elements
    if not resp.elements:
        print("No elements found in the response.")
    else:
        pdf_elements = dict_to_elements(resp.elements)
        # print(f"Extracted {len(pdf_elements)} elements.")
except SDKError as e:
    print(f"Error during partition: {e}")


# Extract tables
tables = [el for el in pdf_elements if el.category == "Table"]
if not tables:
    print("No tables found.")
else:
    print(f"Found {len(tables)} tables.")
    table_html = tables[0].metadata.text_as_html
    # print(table_html)

    # Parse the table's HTML content
    parser = etree.XMLParser(remove_blank_text=True)
    file_obj = StringIO(table_html)
    tree = etree.parse(file_obj, parser)

    # Convert the parsed HTML table into a DataFrame
    df = pd.read_html(etree.tostring(tree, pretty_print=True).decode())[0]
    
    # Display the DataFrame (You can save it to an HTML file)
    df_html = df.to_html(index=False)

    # Save the DataFrame as an HTML file for visualization
    with open('visualized_table.html', 'w') as f:
        f.write(df_html)

    print("Table visualized and saved to 'visualized_table.html'.")

# Continue with other processing if necessary...


### Load the Documents into the Vector DB

elements = chunk_by_title(pdf_elements)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

# Convert elements into documents with metadata
documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Setup retriever
retriever = vectorstore.as_retriever(
    search_type="similarity"
    # search_kwargs={"k": 1}
)

# Setup the Prompt template for answering questions
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

template = """You are an AI assistant for answering questions about the AERB document.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the document, politely inform them that you are tuned to only answer questions about the document.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])

llm = OpenAI()

# Setup the chain
doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
question_generator_chain = LLMChain(llm=llm, prompt=prompt)
qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_chain,
    combine_docs_chain=doc_chain,
)

# # Invoke a question

response = qa_chain.invoke({
    "question": "what is the allowable silvering window for 13-18 duration?",
    "chat_history": []
})["answer"]

print(response)
