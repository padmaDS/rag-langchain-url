import os
import PyPDF2
import docx
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAIChat
 
# Function to read PDF files
def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""
 
# Function to read DOCX files
def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""
 
# Function to split text into chunks
def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
 
# Function to load documents from a specified folder
def load_documents(folder_path):
    documents = []
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return documents
 
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            text = read_pdf(file_path)
            documents.extend(chunk_text(text))  # Add chunked text
        elif filename.endswith('.docx'):
            text = read_docx(file_path)
            documents.extend(chunk_text(text))  # Add chunked text
    return documents
 
# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = ""  # Replace with your actual OpenAI API key  
 
# Load documents from a specified folder
folder_path = "data"  # Ensure this points to the correct directory
documents = load_documents(folder_path)
 
# Check if any documents were loaded
if not documents:
    print("No documents loaded. Please check the folder path.")
else:
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(documents, embeddings)
 
    # Initialize the LLM using OpenAIChat for chat models
    llm = OpenAIChat(model="gpt-4o")  # or "gpt-3.5-turbo"
 
    # Function to answer user questions
    def answer_question(question):
        # Retrieve relevant contexts
        relevant_contexts = vector_store.similarity_search(question)
        if not relevant_contexts:
            return "No relevant documents found."
       
        # Limit the number of contexts retrieved
        max_contexts = 3  # You can adjust this number based on your needs
        relevant_contexts = relevant_contexts[:max_contexts]
 
        # Combine contexts into a single string with truncation if necessary
        context = "\n".join([doc.page_content for doc in relevant_contexts])
       
        # Optional: Truncate context to a certain character limit if needed
        max_length = 2000  # Set your preferred maximum length for the context
        if len(context) > max_length:
            context = context[:max_length] + "..."
 
        # Create the complete prompt
        prompt = (
            "You are a helpful assistant.\n"
            f"User Question: {question}\n"
            f"Context:\n{context}\n"
        )
       
        # Generate an answer based on the question and context
        response = llm.generate(prompts=[prompt])
       
        # Extract the response text
        return response.generations[0][0].text  # Accessing the first generation in the first element of the list
 
    # Example usage
    if __name__ == "__main__":
        while True:
            user_question = input("Please enter your question (or type 'exit' to quit): ")
            if user_question.lower() == 'exit':
                print("Exiting the program.")
                break
           
            answer = answer_question(user_question)
            print("\nAnswer:\n", answer)  # Print the answer on the next line