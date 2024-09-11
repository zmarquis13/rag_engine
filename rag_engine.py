
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory

import markdown2
from bs4 import BeautifulSoup

from dotenv import load_dotenv
import os
import shutil

load_dotenv()

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

def main():
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
    
    
    # Interactive loop
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
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
            result = qa_chain({"question": query})
            print(f"Response: {markdown_to_text(result['answer'])}\n")
