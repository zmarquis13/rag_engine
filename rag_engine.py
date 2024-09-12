from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory

from IPython.display import Markdown
from tqdm import tqdm
import os
import shutil

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import numpy as np

from dotenv import load_dotenv

load_dotenv()

# Brave Search API configuration
BRAVE_SEARCH_API_KEY = os.environ.get("BRAVE_SEARCH_API_KEY")
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# OpenAI API configuration
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# 1. Web Search Integration using Brave Search
def brave_search(query, count=5):
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY
    }
    params = {
        "q": query,
        "count": count
    }
    response = requests.get(BRAVE_SEARCH_ENDPOINT, headers=headers, params=params)
    return response.json()['web']['results']

# 2. Web Scraping and Content Extraction
def extract_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

# 3. Text Processing
def preprocess_text(text, max_length=10000):
    # Simple preprocessing: truncate to max_length characters
    return text[:max_length]

# 4. Embedding Generation using OpenAI
def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 5. Retrieval
def retrieve_relevant_info(query_embedding, doc_embeddings, docs, top_k=3):
    similarities = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [docs[i] for i in top_indices]

# 6 & 7. Prompt Engineering and Language Model Integration
def generate_answer(history, query, relevant_info):
    prompt = f"""Given the following information, please provide a concise and informative answer to the query.

History: {history}

Query: {query}

Relevant Information:
{relevant_info}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate and concise information based on the given query, chat history, and relevant information retrieved."},
            {"role": "user", "content": prompt}
        ]#,
        #max_tokens=
    )
    return response.choices[0].message.content.strip()

# Main RAG function
def rag_web_search(history, query):
    # Perform web search
    search_results = brave_search(query)
    
    # Extract and process content
    docs = []
    for result in search_results:
        content = extract_content(result['url'])
        processed_content = preprocess_text(content)
        docs.append(processed_content)
    
    # Generate embeddings
    doc_embeddings = [generate_embedding(doc) for doc in docs]
    query_embedding = generate_embedding(query)
    
    # Retrieve relevant information
    relevant_info = retrieve_relevant_info(query_embedding, doc_embeddings, docs)
    
    # Generate final answer
    answer = generate_answer(history, query, "\n".join(relevant_info))
    
    return answer

def markdown_to_text(markdown_content):
    # Convert Markdown to HTML
    html_content = markdown2.markdown(markdown_content)
    
    # Parse HTML and extract plain text
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

#create a chromadb (vectorized database for RAG) from a given directory of pdf files
def create_db_from_pdf_directory(pdf_directory, db_directory):
    # initialize an empty list to store all documents
    all_documents = []
    
    # iterate through all files in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            #print(f"Processing {filename}...")
            
            # load pdf
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # add documents to the list
            all_documents.extend(documents)
    
    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    # vector embeddings instance
    embeddings = OpenAIEmbeddings()
    
    # create chromadb
    db = Chroma.from_documents(texts, embeddings, persist_directory=db_directory)
    
    print(f"Database created successfully at {db_directory}")
    return db


#specify source and database and create database with it
pdf_directory = "pdfsnew_"

db_directory = "chromadb"

if not os.path.exists(db_directory):
    #print("Creating new Chroma database...")
    db = create_db_from_pdf_directory(pdf_directory, db_directory)
else:
    #print("Loading existing Chroma database...")
    embeddings = OpenAIEmbeddings()
    db = Chroma(embedding_function=embeddings)
    #print("Database loaded")


def query_db(db, query):
    # Create a retriever from the database
    retriever = db.as_retriever()
    
    # Create a ChatOpenAI model
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Add memory to keep track of conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a Conversational Retrieval Chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(chat, retriever=retriever, memory=memory)
    
    # Run the query 
    result = qa_chain({"memory": memory, "question": query})
    
    return result['answer']

def digest_new_pdfs():
    new_pdf_dir = 'new_pdfs'
    used_pdf_dir = 'used_pdfs'

    added_documents = []

    if os.listdir(new_pdf_dir):

        print('processing new content')
        for file in os.listdir(new_pdf_dir):
            if file.endswith(".pdf"):
                # load pdf
                pdf_path = os.path.join(new_pdf_dir, file)
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # add documents to the list
                added_documents.extend(documents)
                
                source_file = os.path.join(new_pdf_dir, file)
                destination_file = os.path.join(used_pdf_dir, file)
                shutil.move(source_file, destination_file)
    
        # split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(added_documents)
        
        # vector embeddings instance
        embeddings = OpenAIEmbeddings()
    
        # add the embeddings to the existing ChromaDB instance
        db.add_documents(documents=texts, embeddings=embeddings)

        print('processing complete')

# First digest new pdfs
digest_new_pdfs()

# Create a retriever from the database
retriever = db.as_retriever()

# Create a ChatOpenAI model
chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Add memory to keep track of conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a Conversational Retrieval Chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(chat, retriever=retriever, memory=memory)

history = ''

# Interactive loop
while True:
    query = input("\nAsk a question (or type 'exit' to exit): ")
    if query.lower() == 'exit':
        break
    elif query.lower() == 'add':
        # Add new PDFs to ChromaDB
        new_pdf_directory = input("Enter the path to the directory with new PDFs: ")
        # Function to add new PDFs would go here
        # add_new_pdfs_to_chromadb(new_pdf_directory, db)
        print("New PDFs added to ChromaDB.")
    else:
        #response = query_db(db, query)
        
        # Run the query 
        modified_query = query + " use textual evidence where beneficial and put dollar signs on each side of any equations used"
        result = qa_chain({"question": modified_query})
        answer = result['answer']
        
        if (answer == "I don't know."):
            print('No answers found in context library. Searching the web: ')
            answer = rag_web_search(history, query + " cite specific source links used and put dollar signs on each side of any latex expressions or equations used")
            
        history += f"query: {query}\n answer: {answer}\n"
        display(Markdown("**Response:** " + answer))
