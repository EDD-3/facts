import langchain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from redundant_filter_retriever import RedundantFilterRetriever

# Debug line to see verbose info for chain process
langchain.debug = True

load_dotenv()
chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

# Creating an instance of chroma
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,
)

retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# Creating a chain for "Question and answers" interaction
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    # Stuff as in stuffing a turkey
    chain_type="stuff",
)

result = chain.run("What is an interesting fact about the English language?")

print(result)
