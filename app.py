import os
from flask import Flask, render_template, request, session
import gunicorn
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"
app.secret_key = 'your_secret_key'  # Set your secret key for session management

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'Yourapikey'

def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=app.config['UPLOAD_FOLDER'])
    vectordb.persist()

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    return qa

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'pdf_file_path' in session:
        pdf_file_path = session['pdf_file_path']
    else:
        pdf_file_path = None

    if request.method == 'POST':
        # Check if the file is uploaded for the first time
        if 'pdf_file' in request.files and request.files['pdf_file'].filename != '':
            file = request.files['pdf_file']
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                session['pdf_file_path'] = file_path
                pdf_file_path = file_path

        # Get the prompt from the form
        prompt_text = request.form['prompt_text']

        # Load the PDF and create the QA model if not done already
        if pdf_file_path and not hasattr(app, 'qa_model'):
            app.qa_model = load_pdf(pdf_file_path)

        # Get the result from the QA model
        result = app.qa_model(prompt_text)["result"] if hasattr(app, 'qa_model') else None

        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
