### chatwith_multiple_pdfs_using_gemini_API
# Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#load dotenv
load_dotenv()
#set the google api key
os.getenv('GooGLE_API_KEY')
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#get the list of pdf files through function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with open(pdf, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    return text
# Function to split text into chunks
def split_text_into_chunks(text, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks
# Function to create embeddings and vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings=embeddings)
    vector_store.save_local('faiss_index')

#funcation to get conversation chats
def get_conversation_chats():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model=ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt= PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get user input question 
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model='model/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chats()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.wtrite("Answer: ", response['output_text'])

# Main function to run the Streamlit app
def main():
    st.title("Chat with Multiple PDFs using Gemini API")
    st.write("Upload multiple PDF files to chat with them.")
    
    # File uploader for multiple PDF files
    pdf_docs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if pdf_docs:
        # Get text from the uploaded PDFs
        text = get_pdf_text(pdf_docs)
        
        # Split the text into chunks
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000)
        chunks = split_text_into_chunks(text, chunk_size)
        
        # Create vector store
        create_vector_store(chunks)
        
        # User input for question
        user_question = st.text_input("Ask a question about the PDFs:")
        
        if user_question:
            user_input(user_question)
# Run the main function
if __name__ == "__main__":
    main()