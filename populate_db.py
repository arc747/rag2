from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms import ollama
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from typing import List, Dict
from embeddings import generate_embeddings
import re, os, shutil, argparse


CHROMA_PATH ="chroma"
DATA_PATH="docs"




def load_documents(data_path:str) -> Document:
    doc_loader = PyPDFDirectoryLoader(path=data_path)
    docs = doc_loader.load()
    return docs



def text_splitter(docs: Document) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=150,
        is_separator_regex=False,
        length_function=len
        )
    chunks = text_splitter.split_documents(documents=docs)
    return chunks


def add_to_chroma(chunks: List[Document]) -> None:
    
    chunks_with_ids = get_chunk_ids(chunks)
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function = generate_embeddings())
    
    existing_items = db.get()
    existing_sources = existing_items.get("metadatas")
    existing_ids = set(existing_items.get("ids"))
    print(f"Number of docs already in db: {len(existing_ids)} from documents {existing_sources}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            

    if new_chunks:
        print(f"Total new chunks to add: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(documents=new_chunks, ids=new_chunk_ids)
        db.persist()
        print(f"New chunks added with ids:\n{new_chunk_ids}")
    else:
        print("Nothing new to add..")

    return existing_ids


def get_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_source = None
    last_page = None
    for chunk in chunks:
        current_source = re.sub(r"^docs\\|\.pdf$", "", chunk.metadata.get("source", "").strip())
        current_page = str(chunk.metadata.get("page", "")).strip()
        if (current_page!=last_page)|(current_source!=last_source):
            num = 1
            # chunk.metadata["id"] = f"{source}:{current_page}:{num}"
        else:
            num += 1
        chunk.metadata["id"] = f"{current_source}:{current_page}:{num}"
        last_page = current_page
        last_source = current_source 
    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="clear the db")
    args = parser.parse_args()
    print(type(parser))
    print(type(args.reset))
    if args.reset:
        print("Clearing the db..🔥🔥")
        clear_database()
        
    docs = load_documents(data_path="docs")
    chunks = text_splitter(docs=docs)
    add_to_chroma(chunks)
    

if __name__=="__main__":
    main()

    

