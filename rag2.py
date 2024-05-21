from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from embeddings import generate_embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from typing import List, Dict
import argparse


class Rag2:

    def __init__(self, chroma_path: str) -> None:
        self.chroma_path = chroma_path
        self.prompt_template = """
Context: {context}\n
Answer the questions based on the above given context. 
Question: {query}
                               """


    def query_rag(self, query_text: str) -> List[Document]:
        """
        Query the vectorstore based on similarity score to create a context
        """
        db = Chroma(persist_directory=self.chroma_path, embedding_function=generate_embeddings())
        rag_results = db.similarity_search_with_score(query=query_text, k=10)
        if rag_results:
            context = rag_results
        else:
            context = None
        return context


    def prep_prompt(self, query_text):
        """
        Create the prompt with context to query the model
        """
        rag_context = self.query_rag(query_text)
        if rag_context:
            self.context = [doc.page_content for doc, score in rag_context]
            self.source = [dict(doc.metadata, score=score) for doc, score in rag_context]
        else:
            self.context = None
            self.source = None
        prompt_template = ChatPromptTemplate.from_template(template=self.prompt_template)
        llm_prompt = prompt_template.format(context=self.context, query=query_text)
        return llm_prompt
        

    def query_llm(self, query_text):
        """
        Query the model to generate a rag response.
        """
        llm_prompt = self.prep_prompt(query_text=query_text)
        llm = Ollama(model= "llama3")
        llm.temperature=0
        # response = llm.invoke(llm_prompt)
        chain = llm | StrOutputParser()
        response = chain.invoke(llm_prompt)
        # response = StrOutputParser(inputs=llm_response)
        return (response, self.source if self.source is not None else "No context found")
        
    
CHROMA_PATH = "chroma"
rag2 = Rag2(chroma_path=CHROMA_PATH)

def main():
    parser = argparse.ArgumentParser()
    print("-------")
    parser.add_argument("query_text", type=str, help="Query for the LLM")
    args = parser.parse_args()
    query_text = args.query_text
    # response = rag2.query_rag(query_text=query_text)
    response = rag2.query_llm(query_text=query_text)
    print(response[0], response[1])
    

if __name__=="__main__":
    main()
    
    
    
