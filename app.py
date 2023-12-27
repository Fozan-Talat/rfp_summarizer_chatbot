# import azure.functions as func
import logging
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from io import BytesIO
import json
import re
from flask import Flask, jsonify, request
 
# Import Azure OpenAI
from langchain.chat_models import AzureChatOpenAI
import os
 
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
 
# OPENAI_DEPLOYMENT_NAME = "ksapakflask"
# OPENAI_EMBEDDINGS_NAME = "ksapakflaskemb"
# OPENAI_API_VERSION = "2023-07-01-preview"
# OPENAI_API_KEY = "82d9782d4c2a43e797e16b2e20908bda"
# OPENAI_API_BASE = "https://ksapakflask.openai.azure.com/"
# OPENAI_API_TYPE = "azure"
# MODEL_TEMPERATURE = 0

OPENAI_DEPLOYMENT_NAME = "ksapak"
OPENAI_EMBEDDINGS_NAME = "ksapakemb"
OPENAI_API_VERSION = "2023-07-01-preview"
OPENAI_API_KEY = "1a6fd35bc9dc46f88219a7cb4639f037"
OPENAI_API_BASE = "https://ksapoc.openai.azure.com/"
OPENAI_API_TYPE = "azure"
MODEL_TEMPERATURE = 0
 
vectorstore = None
conversation_chain = None
 
 
llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      temperature=MODEL_TEMPERATURE,
                      openai_api_version=OPENAI_API_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_base=OPENAI_API_BASE,
                      openai_api_type=OPENAI_API_TYPE)
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
word_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
 
def get_pdf_text(pdf_data):
    pdf_bytes = BytesIO(pdf_data)
    pdf_reader = PdfReader(pdf_bytes)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
   
    return pdf_text
 
 
# ******************************* functions for summary *******************************
 
def get_token_count(text):
    text_token_count = word_splitter.count_tokens(text=text.replace("\n"," "))
    return text_token_count
 
def create_single_doc(text):
    return Document(page_content=text)
   
def create_multiple_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=3900, chunk_overlap=100, length_function=get_token_count)
    docs = text_splitter.create_documents([text])
    return docs
 
def create_docs(text):
    if get_token_count(text) > 3900:
        return create_multiple_docs(text)
    else:
        return [create_single_doc(text)]
 
def generate_summary(docs):
 
    prompt_template = """Please summarize the key information from the attached RFP documents.
    Highlight the main objectives, requirements, evaluation criteria, and any specific terms or
    conditions that are essential for understanding and responding to this RFP.
    TEXT: {text}
    SUMMARY:
    """
    prompt = PromptTemplate.from_template(prompt_template)
 
    refine_template="""Write a concise summary of the following text delimited by triple backquotes.
    Return your response in a paragraph which covers the key points of the text.
    ```{text}```
    PARAGRAPH SUMMARY:
    """
 
    refine_prompt = PromptTemplate.from_template(refine_template)
 
    refine_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        verbose=True
    )
 
    refine_outputs = refine_chain({"input_documents": docs})
 
    # print(refine_outputs["output_text"])
    return (refine_outputs["output_text"])
 
 
# ******************************* functions for Q&A *******************************
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=8000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
 
 
def get_vectorstore(text_chunks):
    global vectorstore
    # embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(deployment=OPENAI_EMBEDDINGS_NAME,
                                  openai_api_version=OPENAI_API_VERSION,
                                  openai_api_key=OPENAI_API_KEY,
                                  openai_api_base=OPENAI_API_BASE,
                                  openai_api_type=OPENAI_API_TYPE, chunk_size=1)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
 
def get_conversation_chain(vectorstore):
    global conversation_chain
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      temperature=MODEL_TEMPERATURE,
                      openai_api_version=OPENAI_API_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_base=OPENAI_API_BASE,
                      openai_api_type=OPENAI_API_TYPE)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
 
 
def qna(conversation_chain, question):
    answer = conversation_chain({"question": question})["answer"]
    print("Answer:", answer)
    return answer
 
def get_title(conversation_chain):
    question = f"""
    Extract the title from the RFP document:
    Title:
    """
    answer = conversation_chain({"question": question})["answer"]
    print("Answer:", answer)
    return answer
 
def get_submission_date(conversation_chain):
    question = f"""
    Extract the submission date from the RFP document:
    Submission Date:
    """
    answer = conversation_chain({"question": question})["answer"]
    print("Answer:", answer)
    return answer
 
def get_submission_address(conversation_chain):
    question = f"""
    Extract the submission address from the RFP document:
    Submission Address:
    """
    answer = conversation_chain({"question": question})["answer"]
    print("Answer:", answer)
    return answer
 
 
def get_date_field(text):
    # Schema
    schema = {
        "properties": {
            "date": {"type": "string"},
            # "date": {"type": "string"}
        },
        "required": ["date"],
    }
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      temperature=MODEL_TEMPERATURE,
                      openai_api_version=OPENAI_API_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_base=OPENAI_API_BASE,
                      openai_api_type=OPENAI_API_TYPE)
    chain = create_extraction_chain(schema, llm)
    # print("abc")
    ans = chain.run(text)
 
    field = ans[0]["date"]
 
    if field == "submission date":
        return "not mentioned"
    else:
        return  field
 
def get_address_field(text):
    # Schema
    schema = {
        "properties": {
            "address": {"type": "string"},
            # "date": {"type": "string"}
        },
        "required": ["address"],
    }
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                      temperature=MODEL_TEMPERATURE,
                      openai_api_version=OPENAI_API_VERSION,
                      openai_api_key=OPENAI_API_KEY,
                      openai_api_base=OPENAI_API_BASE,
                      openai_api_type=OPENAI_API_TYPE)
    chain = create_extraction_chain(schema, llm)
    # print("abc")
    ans = chain.run(text)
 
    field = ans[0]["address"]
 
    if field == "submission address":
        return "not mentioned"
    else:
        return  field
 
app = Flask(__name__)
# ******************************* API Routes *******************************
 
# app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
@app.route('/api/http_summarizer', methods=['POST'])
def http_summarize():
    logging.info('Python HTTP trigger function processed a request.')
 
    if "file" in request.files:
        file = request.files['file']
        file_contents = file.read()
 
        text = get_pdf_text(file_contents)
 
        # Summary generation
        single_doc = create_single_doc(text)
        print("token count of single doc: ", get_token_count(single_doc.page_content))
        docs = create_docs(text)
        for doc in docs:
            print(get_token_count(doc.page_content))
 
        # Initialization for QnA
        print("****************************************************************************")
 
        # Get text chunks
        text_chunks = get_text_chunks(text)
 
        # Create vector store
        vectorstore = get_vectorstore(text_chunks)
 
        # Create conversation chain
        conversation_chain = get_conversation_chain(vectorstore)
 
        summary = generate_summary(docs)
        summary = re.sub(r'-\s|\n\n?|\n', ' ', summary)     # Remove bullets and new lines
        title = get_title(conversation_chain)
        if "not provided" in title.lower() or "not mentioned" in title.lower():
            title = "No title available"
        title = re.sub(r'^.*?is\s+', '', title)
       
        submission_date = get_date_field(get_submission_date(conversation_chain))
        submission_address = get_address_field(get_submission_address(conversation_chain))
 
        # Create a JSON response
        response_data = {"summary": summary,
                         "title":title,
                         "submission_date":submission_date,
                         "submission_address":submission_address}
 
        return jsonify(response_data)
    else:
        return jsonify({"error": "No file attached"}), 400
 
 
   
@app.route('/api/http_qna', methods=['POST'])
def http_qna():
    logging.info('Python HTTP trigger function processed a request.')
    # Extract the question from the request
    question = request.form.get("question")
 
    # QnA
    query_ans = qna(conversation_chain, question)
 
    # Create a JSON response
    response_data = {"answer": query_ans}
 
    # Return the file contents in the response
    return jsonify(response_data)
 
if __name__ == '__main__':
    app.run(debug=True)