

%pip install ../../../amazon-bedrock-workshop/dependencies/botocore-1.29.162-py3-none-any.whl ../../../amazon-bedrock-workshop/dependencies/boto3-1.26.162-py3-none-any.whl ../../../amazon-bedrock-workshop/dependencies/awscli-1.27.162-py3-none-any.whl --force-reinstall

import boto3
import json
import os
import sys

module_path = "../../../amazon-bedrock-workshop/"
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
boto3_bedrock = bedrock.get_bedrock_client(os.environ.get('BEDROCK_ASSUME_ROLE', None))

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain

import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


cl_llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={"max_tokens_to_sample": 300,
                                                                                     "temperature":0.9,
                                                                                    "top_k":300,
                                                                                    "top_p":0.99})

def generate_response(txt):
    # Instantiate the LLM model
    llm = cl_llm
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    response = generate_response(txt_input)
    result.append(response)
        
if len(result):
    st.info(response)
