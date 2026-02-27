# Youtube_Transcrip_RAG_System

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about any YouTube video by intelligently retrieving answers from the video's transcript.

---

## 📌 Overview

This project fetches the transcript of a YouTube video, processes it into searchable chunks, and uses OpenAI's language model to answer user questions strictly based on the video content — no hallucinations, no outside knowledge.

---

## 🧠 How It Works

```
YouTube Video ID
      ↓
Fetch Transcript (YouTube Transcript API)
      ↓
Split into Chunks (RecursiveCharacterTextSplitter)
      ↓
Generate Embeddings (OpenAI text-embedding-3-small)
      ↓
Store in FAISS Vector Store
      ↓
User Question → Retrieve Top-K Relevant Chunks
      ↓
GPT-4o-mini generates answer from context
```

---

## 🛠️ Tech Stack

| Component        | Tool / Library                  |
|------------------|---------------------------------|
| Language         | Python                          |
| LLM              | OpenAI GPT-4o-mini              |
| Embeddings       | OpenAI text-embedding-3-small   |
| Vector Store     | FAISS                           |
| RAG Framework    | LangChain                       |
| Transcript API   | youtube-transcript-api          |
| Config           | python-dotenv                   |

---

## 📁 Project Structure

```
youtube-qa/
│
├── chains.py          # Main RAG pipeline
├── .env               # API keys (not committed)
├── requirements.txt   # Dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/youtube-qa.git
cd youtube-qa
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the project
```bash
python chains.py
```

---

## 📦 Requirements

```txt
langchain
langchain-openai
langchain-community
faiss-cpu
youtube-transcript-api
python-dotenv
```

---

## 💡 Usage

1. Open `chains.py` and set your desired YouTube video ID:
```python
video_id = "YOUR_VIDEO_ID_HERE"
```

2. Change the question at the bottom of the file:
```python
r = main_chain.invoke("Your question about the video here")
print(r)
```

3. Run the script and get your answer!

---

## 🔒 Notes

- The model only answers from the video transcript. If the answer isn't in the transcript, it will say so.
- Only YouTube videos with English captions (`en`) are supported.
- Your OpenAI API key is required and is never hardcoded — always use `.env`.

---

## 🙋‍♂️ Author

**Tushar Mishra**  
tusharmish25@gmail.com
