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
    return ChatOpenAI(temperature=0.1, model="gpt-4o")

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
    return process_feature(content, "provide a brief review of the background and previous research related to the paper, including clickable references for a technical postdoctoral scientific/engineering audience. Also provide definitions for technical terms or jargon used in the paper.", chat_model)

def generate_critical_review(content, chat_model):
    return process_feature(content, "provide a critical review of the paper for a technical postdoctoral scientific/engineering audience, pointing out potential weaknesses in experimental design, data interpretation, or statistical analysis. You are allowed be overly harsh.", chat_model)

def generate_discussion_questions(content, chat_model):
    return process_feature(content, "generate 20 thought-provoking questions based on the content of the paper for a technical postdoctoral scientific/engineering audience, encouraging deeper analysis of methodology, results, and implications.", chat_model)

def analyze_figures(content, images, chat_model):
    image_analysis = process_feature(content, "provide an interpretation of the key data points and trends in the figures, graphs, and tables mentioned in the paper for a technical postdoctoral scientific/engineering audience.", chat_model)
    return image_analysis, images

def handle_file_upload(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()

        full_text = "\n".join([page.page_content for page in pages])

        return full_text

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")  # Set the page layout to wide
    st.title("Journal Club Scientific Article Analysis App")

    if 'full_text' not in st.session_state:
        st.session_state.full_text = None

    # Sidebar for file upload
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your scientific article (PDF)", type="pdf")

        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                full_text = handle_file_upload(uploaded_file)
                if full_text:
                    st.session_state.full_text = full_text
                    st.success("PDF processed successfully!")
                else:
                    st.error("An error occurred while processing the PDF.")

    # Main content area
    if st.session_state.full_text:
        st.subheader("Analysis Features")
        st.write("Select a feature to analyze the paper:")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            background_context = st.button("Background Context")
        with col2:
            figure_analysis = st.button("Figure Analysis")
        with col3:
            critical_review = st.button("Critical Review")
        with col4:
            discussion_questions = st.button("Discussion Questions")

        # Create a wide container for output
        output_container = st.container()

        with output_container:
            if background_context:
                with st.spinner("Generating background context..."):
                    result = generate_background_context(st.session_state.full_text, create_chat_model())
                    st.markdown(result)

            elif figure_analysis:
                with st.spinner("Analyzing figures..."):
                    result, figures = analyze_figures(st.session_state.full_text, extract_images_from_pdf(uploaded_file), create_chat_model())
                    st.write(result)

            elif critical_review:
                with st.spinner("Generating critical review..."):
                    result = generate_critical_review(st.session_state.full_text, create_chat_model())
                    st.write(result)

            elif discussion_questions:
                with st.spinner("Generating discussion questions..."):
                    result = generate_discussion_questions(st.session_state.full_text, create_chat_model())
                    st.write(result)

if __name__ == "__main__":
    main()
