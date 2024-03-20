from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()
chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

# Creating an instance of chatgpt
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

retriever = db.as_retriever()

# Creating a chain for QA interaction
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    # Stuff as in stuffing a turkey
    chain_type="stuff",
)

result = chain.run("What is an interesting fact about the English language?")

print(result)
