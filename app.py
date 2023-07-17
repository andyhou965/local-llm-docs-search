import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMADB_SETTING

# model and tokenizer loading
checkpoint = "llms/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, device_map='auto', torch_dtype=torch.float32
)


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=CHROMADB_SETTING.persist_directory,
        embedding_function=embeddings,
        client_settings=CHROMADB_SETTING,
    )
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    # metadata = generated_text['metadata']
    # for text in generated_text:

    #     print(answer)

    # wrapped_text = textwrap.fill(response, 100)
    # return wrapped_text
    return answer, generated_text


def main():
    st.title("Search Your Documents 📁")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI app that allows you to search your documents using natural language.
            """
        )
    question = st.text_area("Enter Your Question:")
    if st.button("Search"):
        st.info("Your question: " + question)
        st.info("Your Answer")
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write(metadata)


if __name__ == "__main__":
    main()
