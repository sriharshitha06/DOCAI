from src.helper import load_pdf_file, text_split ,download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='Data/')
text_chunks =  text_split(extracted_data)
embeddings= download_hugging_face_embeddings()

pc = Pinecone(api_key="pcsk_UB8WB_H1wzMgzqJ73hQUfRuQQpzaTRHdiRnQJTQFVF6hciNEkodaBGe4XzEFiW37LDQNr")
index_name = "docai"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
#Embbed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name = "docai",
    embedding=embeddings,

)
#embed each chunk and upsert the embeddings into your pinecone index 
docsearch = PineconeVectorStore.from_existing_index(
    index_name= "docai",
    embedding = embeddings

)
