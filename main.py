import os
from dotenv import load_dotenv, find_dotenv

# depracated
# from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_huggingface.llms import HuggingFacePipeline

load_dotenv(find_dotenv())
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")


model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    clean_up_tokenization_spaces=True,
)
llm = HuggingFacePipeline(pipeline=pipe)

doc_loader = TextLoader("./jinnah.txt")
document = doc_loader.load()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "."], chunk_size=500, chunk_overlap=80
)
splitted_documents = splitter.split_documents(document)

hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACE_ACCESS_TOKEN,
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)

db = FAISS.from_documents(splitted_documents, hf_embeddings)

template = """
Answer the following question based only on the provided context:

Context: {context}

Question: {input}
"""
prompt = PromptTemplate.from_template(template=template, template_format="f-string")

combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(db.as_retriever(), combine_docs_chain)

while True:
    question = input("Question : ")
    answer = retrieval_chain.invoke({"input": question})
    print("Answer : ", answer["answer"])
