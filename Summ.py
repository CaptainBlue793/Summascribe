"""TEXT SUMMARIZATION Web APP"""

#Importing Packages
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

#Model and Tokenizer
checkpoint="Lamini-1"
tokenizer=T5Tokenizer.from_pretrained(checkpoint)
base_model=T5ForConditionalGeneration.from_pretrained(checkpoint,device_map='auto',torch_dtype=torch.float32)

#File Loader & Processing
def file_processing(file):
    loader=PyPDFLoader(file)
    pages=loader.load_and_split()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
    texts=text_splitter.split_documents(pages)
    final_texts=""
    for text in texts:
        print(text)
        final_texts=final_texts+text.page_content
    return final_texts

#Language Model Pipeline
def llm_pipeline(filepath):
    pipe_summ=pipeline('summarization',model=base_model,tokenizer=tokenizer,max_length=500,min_length=50)
    input=file_processing(filepath)
    result=pipe_summ(input)
    result=result[0]['summary_text']
    return result

#Streamlit Code
st.set_page_config(layout='wide')

#Display Background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('wallpapersden.com_minimalistic-dark-hills-blue_3840x2160.jpg')

#Font Style
with open('font.css') as f:
    st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)

#Sidebar
st.sidebar.image('asw.png')
st.sidebar.title("ABOUT THE APP")
st.sidebar.write("Summascribe is a web application that simplifies PDF summarization. It uses Streamlit and LangChain "
                "to generate concise summaries by applying advanced NLP algorithms. ")
st.sidebar.write("The app quickly processes the text, understands context, and generates coherent "
          "summaries. Summascribe is a valuable tool for extracting essential information from PDFs.")
#Display pdf of a given file
@st.cache_data
def display(file):
    #Opening file from filepath
    with open(file,"rb") as f:
        base64_pdf=base64.b64encode(f.read()).decode('utf-8')
    #Embedding pdf in html
    display_pdf=F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" ' \
                F'type="application/pdf"></iframe>'
    #Displaying File
    st.markdown(display_pdf,unsafe_allow_html=True)

def main():
    st.title("SUMMASCRIBE")
    st.subheader('Text Document Summarization using Large Language Models')
    uploaded_file=st.file_uploader("Upload PDF file",type=['pdf'])
    with st.expander("NOTE"):
        st.write(
            "Summascribe currently accepts PDF documents that contain only text and no images. This limitation is due to our app's current focus on leveraging advanced natural language processing (NLP) algorithms to extract key information from textual content.")
    if uploaded_file is not None:
        if st.button("Summarize"):
            col1,col2=st.columns((3,2))
            filepath="data/"+uploaded_file.name
            with open(filepath,'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info('Uploaded File')
                pdf_viewer=display(filepath)
            with col2:
                st.spinner(text="In progress...")
                st.info('Summary')
                summary=llm_pipeline((filepath))
                st.success(summary)
if __name__=='__main__':
    main()
