from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMADB_SETTING

persist_directory = CHROMADB_SETTING.persist_directory


def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(os.path.join(root, file))
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    # Create an embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create a vector store
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMADB_SETTING,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    main()
