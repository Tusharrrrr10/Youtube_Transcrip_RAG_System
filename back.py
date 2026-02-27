from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class LoadRequest(BaseModel):
    video_id: str

class AskRequest(BaseModel):
    question: str

# Fixed objects — initialized once
ytt_api = YouTubeTranscriptApi()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer only from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

main_chain = None


@app.post("/load")
def load_video(body: LoadRequest):
    global main_chain

    try:
        transcript_list = ytt_api.fetch(body.video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="No captions available for this video")

    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    main_chain = parallel_chain | prompt | llm | parser

    return {"status": "ok", "chunks": len(chunks)}


@app.post("/ask")
def ask(body: AskRequest):
    if main_chain is None:
        raise HTTPException(status_code=400, detail="No video loaded yet")

    r = main_chain.invoke(body.question)
    return {"answer": r}