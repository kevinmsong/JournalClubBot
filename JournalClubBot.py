import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import PyPDF2
import fitz  # PyMuPDF
import io
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key
api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = api_key

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def create_chat_model():
    return ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_images_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return images

def process_feature(content, feature, chat_model):
    system_message = SystemMessage(content=f"You are an AI assistant analyzing a scientific article. Your task is to {feature}")
    human_message = HumanMessage(content=f"Here's the content of the scientific article:\n\n{content}\n\nPlease provide your analysis based on the task.")
    
    response = chat_model([system_message, human_message])
    return response.content

def generate_background_context(content, chat_model):
    return process_feature(content, "provide a brief review of the background and previous research related to the paper, including clickable references. Also provide definitions for technical terms or jargon used in the paper.", chat_model)

def generate_paper_summary(content, chat_model):
    return process_feature(content, "provide a concise summary of the paper, highlighting the main objectives, methods, results, and conclusions.", chat_model)

def generate_critical_review(content, chat_model):
    return process_feature(content, "provide a critical review of the paper, pointing out potential weaknesses in experimental design, data interpretation, or statistical analysis.", chat_model)

def generate_discussion_questions(content, chat_model):
    return process_feature(content, "generate 20 thought-provoking questions based on the content of the paper, encouraging deeper analysis of methodology, results, and implications.", chat_model)

def analyze_figures(content, images, chat_model):
    image_analysis = process_feature(content, "provide an interpretation of the key data points and trends in the figures, graphs, and tables mentioned in the paper.", chat_model)
    return image_analysis, images

def handle_file_upload(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()

        full_text = "\n".join([page.page_content for page in pages])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts, embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=create_chat_model(),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        summary_prompt = f"Please provide a brief summary of the following text, which is the content of the uploaded PDF titled '{uploaded_file.name}':\n\n{full_text[:2000]}"
        summary = create_chat_model().predict(summary_prompt)

        return qa_chain, summary, full_text

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return None, f"An error occurred while processing the PDF: {str(e)}", None

def main():
    st.title("Scientific Article Analysis App")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [SystemMessage(content="You are an AI assistant answering questions about a scientific article.")]
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'full_text' not in st.session_state:
        st.session_state.full_text = None

    uploaded_file = st.file_uploader("Upload your scientific article (PDF)", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            qa_chain, summary, full_text = handle_file_upload(uploaded_file)
            if qa_chain:
                st.session_state.qa_chain = qa_chain
                st.session_state.full_text = full_text
                st.success("PDF processed successfully!")
                st.subheader("Summary")
                st.write(summary)
            else:
                st.error(summary)  # Display error message

    if st.session_state.qa_chain and st.session_state.full_text:
        st.write("Select a feature to analyze the paper:")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        with col1:
            if st.button("Background Context"):
                with st.spinner("Generating background context..."):
                    result = generate_background_context(st.session_state.full_text, create_chat_model())
                    st.markdown(result)

        with col2:
            if st.button("Paper Summary"):
                with st.spinner("Generating paper summary..."):
                    result = generate_paper_summary(st.session_state.full_text, create_chat_model())
                    st.write(result)

        with col3:
            if st.button("Question Answering"):
                st.write("Question Answering Mode Activated")
                user_input = st.text_input("Your question:")
                if st.button("Submit Question"):
                    if user_input:
                        with st.spinner("Generating answer..."):
                            response = st.session_state.qa_chain({"query": user_input})
                            st.write("Answer:", response['result'])

        with col4:
            if st.button("Figure Analysis"):
                with st.spinner("Analyzing figures..."):
                    result, figures = analyze_figures(st.session_state.full_text, extract_images_from_pdf(uploaded_file), create_chat_model())
                    st.write(result)
                    for i, img in enumerate(figures):
                        st.image(img, caption=f"Figure {i+1}", use_column_width=True)

        with col5:
            if st.button("Critical Review"):
                with st.spinner("Generating critical review..."):
                    result = generate_critical_review(st.session_state.full_text, create_chat_model())
                    st.write(result)

        with col6:
            if st.button("Discussion Questions"):
                with st.spinner("Generating discussion questions..."):
                    result = generate_discussion_questions(st.session_state.full_text, create_chat_model())
                    st.write(result)

if __name__ == "__main__":
    main()