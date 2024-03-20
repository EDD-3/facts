from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

load_dotenv()
# The embedding creation algorithm
embeddings = OpenAIEmbeddings()


text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

# Reading the content of the txt file
loader = TextLoader("facts.txt")

# Splitting the text in the file into chunks for embeddings
docs = loader.load_and_split(text_splitter=text_splitter)

# Creating the database and calculating the embeddings that are to be stored in the database
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb",
)

results = db.similarity_search_with_score(
    "What is an interesting fact about the English language?", k=1
)

for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)


# for doc in docs:
#     print(doc)
#     print("\n")
