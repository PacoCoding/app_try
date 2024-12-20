#%% Import Streamlit and Libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from docx import Document  # Import the python-docx library

#%% Function to get text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vectorstore(text_chunks):
    # Initialize OpenAI embeddings using the secret API key
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OpenAI_key"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create the conversational chain
def get_conversation_chain(vectrostore):
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0.1,
        openai_api_key=st.secrets["OpenAI_key"]  # Pass the key explicitly
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectrostore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to save response to a Word file

# Function to handle user input
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents first!")
        return

    # Get the response from the conversation chain
    response = st.session_state.conversation({'question': user_question})
    answer = response['answer']  # Assuming response contains an 'answer' key

    # Display the response in Streamlit
    st.write(answer)


# Set Page Configuration
st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':book:')

# Define Main Function
def main():
    # Debugging (optional): Confirm the key is set
    st.write(f"API Key loaded successfully (masked): {st.secrets['OpenAI_key'][:5]}...")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    

    # Input for questions
    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input('Ask a question about your document')
    
    if user_question:
        handle_userinput(user_question)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader('Upload your PDFs here and click on Process', 
                                    accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):  # Tell that the model is working
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # Create our vector store with embeddings
                vectrostore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectrostore)

# Entry Point
if __name__ == '__main__':
    main()
