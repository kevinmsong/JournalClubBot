# JournalClubBot

Streamlit helper for journal club: upload a scientific PDF and generate background context, figure-oriented interpretation, a critical review, or discussion questions with GPT-4o.

Live app: [journalclub.streamlit.app](https://journalclub.streamlit.app)

## Features

- Upload a PDF; extract text via LangChain `PyPDFLoader`
- One-click analyses aimed at a technical postdoctoral audience:
  - Background context (prior research, clickable-style references, jargon definitions)
  - Figure analysis (interpretation of figures/graphs/tables described in the text; also extracts embedded images with PyMuPDF)
  - Critical review (strengths/weaknesses, design, stats)
  - Discussion questions (20 prompts)
- Uses OpenAI `gpt-4o` via LangChain `ChatOpenAI`

## Requirements

See `requirements.txt` (streamlit, openai, langchain, langchain-openai, langchain-community, PyPDF2, PyMuPDF, python-dotenv, pypdf, faiss-cpu).

## Setup

```bash
pip install -r requirements.txt
```

Configure Streamlit secrets:

```toml
openai_api_key = "YOUR_OPENAI_API_KEY"
```

## Usage

```bash
streamlit run JournalClubBot.py
```

Upload a PDF in the sidebar, then use the four analysis buttons in the main area.
