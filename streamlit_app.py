import os
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp"
from typing import List, Optional
import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

import nltk

@st.cache_data
def get_stopwords():
    nltk.download('stopwords')

st.set_page_config(page_title="Chat with a friend on the works of Rabindranath Tagore", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with a friend on the works of Rabindranath Tagore")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Rabindranath Tagore!!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    

    Settings.chunk_size = 1500
    Settings.chunk_overlap = 50
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
    embed_batch_size=20,
    token=st.secrets.hftoken,
    )

    
    Settings.llm = HuggingFaceInferenceAPI(
    model_name="deepseek-ai/DeepSeek-R1-0528",
    token=st.secrets.hftoken,
    provider="auto",  # this will use the best provider available
    system_prompt="""You are an expert on the work of Rabindrath Tagore.
    Answer the question using the provided documents, which contain relevant excerpts from the work of Rabindrath Tagore.
    The context for all questions is the work of Rabindrath Tagore. Whenver possible, include a quotation from the provided excerpts of his work to illustrate your point.
    Respond using a florid but direct tone, typical of an early modernist writer.
    Keep your answers very short: respond in under 100 words.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", verbose=True, streaming=False,
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = ""
        try:
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
        except:
            st.error("We got an error from Google Gemini - this may mean the question had a risk of producing a harmful response. Consider asking the question in a different way.")        
        if response_stream != "":
            with st.spinner("waiting"):
                try:
                    st.write_stream(response_stream.response_gen)
                except Exception as e: 
                    print(e)
                    st.error("We hit a bump - let's try again")
                    try:
                        resp = st.session_state.chat_engine.chat(prompt)[0]
                        st.write(resp)
                    except Exception as e: 
                        print(e)
                        st.error("We got an error from Hugging Face - this can happen for a few different reasons. Consider asking the question in a different way.")
            message = {"role": "assistant", "content": response_stream.response}
            # Add response to message history
            st.session_state.messages.append(message)
