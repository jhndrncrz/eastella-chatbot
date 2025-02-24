from flask import Flask, jsonify, request
from flask.templating import render_template

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader

import sys

app = Flask(__name__)

# Load PDF documents and prepare the question-answering system
loader = PyPDFDirectoryLoader(r"C:\Users\User\OneDrive\Desktop\FLASK_CHATBOT")
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(data)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

# Initialize LlamaCpp model
llm = LlamaCpp(
    streaming=True,
    model_path=r"C:\Users\User\OneDrive\Desktop\Flask_Chatbot\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4096
)

# Create QA chain
QA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))

def generate_EASTella_response(user_query):
    result = QA({'query': user_query})
    return result['result']

@app.route('/home')
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("query")
    if user_input:
        response = generate_EASTella_response(user_input)
        return jsonify({"answer": response})
    return jsonify({"answer": "Sorry, I couldn't understand that."})

if __name__ == '__main__':
  app.run()
