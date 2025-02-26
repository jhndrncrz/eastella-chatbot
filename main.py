import os
import requests
from flask import Flask, jsonify, request, render_template
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

app = Flask(__name__)

# Configure PDF directory dynamically for deployment
dirname = os.path.dirname(__file__)
pdf_dir = dirname  # Assuming PDFs are in 'pdfs' folder

# Ensure that the PDF directory exists
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

# Load PDF documents and prepare the question-answering system
loader = PyPDFDirectoryLoader(
    pdf_dir
)  # Change the loader to dynamically look in 'pdfs'
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(data)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

# hf = HuggingFaceHub(
#     repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
#     task="text-generation",
#     model_kwargs={"temperature": 0.75, "max_length": 500},
# )

hf = HuggingFacePipeline.from_model_id(
    model_id="SmallDoge/Doge-160M-Instruct",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.75, "max_new_tokens": 300},
    model_kwargs={"trust_remote_code": True}
)

llm = hf

# Create QA chain
QA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
)


def generate_EASTella_response(user_query):
    result = QA({'query': user_query})
    return result['result']


@app.route("/home")
@app.route("/")
def home():
    return render_template(
        "home.html"
    )  # Ensure you have home.html in the templates directory


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("query")
    if user_input:
        response = generate_EASTella_response(user_input)
        return jsonify({"answer": response})
    return jsonify({"answer": "Sorry, I couldn't understand that."})


if __name__ == "__main__":
    app.run(debug=True)  # Enable debug for local development
