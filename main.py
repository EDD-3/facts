from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
# The embedding creation algorithm
embeddings = OpenAIEmbeddings()


# Splitting the text in the file into chunks for embeddings
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)


for doc in docs:
    print(doc)
    print("\n")
