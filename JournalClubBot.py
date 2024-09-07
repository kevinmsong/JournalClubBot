import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import PyPDF2
import fitz  # PyMuPDF
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

def create_chat_model():
    return ChatOpenAI(temperature=0.1, openai_api_key=api_key, model="gpt-4-0125-preview")

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

def main():
    st.title("Journal Club Scientific Article Analysis App")

    uploaded_file = st.file_uploader("Upload your scientific article (PDF)", type="pdf")

    if uploaded_file is not None:
        content = extract_text_from_pdf(uploaded_file)
        images = extract_images_from_pdf(uploaded_file)

        chat_model = create_chat_model()

        st.write("Select a feature to analyze the paper:")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Background Context"):
                st.write("Generating background context...")
                result = generate_background_context(content, chat_model)
                st.markdown(result)

            if st.button("Paper Summary"):
                st.write("Generating paper summary...")
                result = generate_paper_summary(content, chat_model)
                st.write(result)

        with col2:
            if st.button("Question Answering"):
                st.write("Initializing question answering...")
                st.session_state.chat_history = st.session_state.get('chat_history', [SystemMessage(content="You are an AI assistant answering questions about a scientific article.")])
                st.session_state.chat_history.append(AIMessage(content="Hello! I'm here to answer questions about the uploaded scientific article. What would you like to know?"))

                for message in st.session_state.chat_history[1:]:  # Skip the system message
                    st.write(f"{'You' if isinstance(message, HumanMessage) else 'AI'}: {message.content}")

                user_input = st.text_input("Your question:")
                if user_input:
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    response = chat_model(st.session_state.chat_history + [HumanMessage(content=f"Article content: {content}")])
                    st.session_state.chat_history.append(AIMessage(content=response.content))
                    st.write(f"AI: {response.content}")

            if st.button("Figure Analysis"):
                st.write("Analyzing figures...")
                result, figures = analyze_figures(content, images, chat_model)
                st.write(result)
                for i, img in enumerate(figures):
                    st.image(img, caption=f"Figure {i+1}")

        with col3:
            if st.button("Critical Review"):
                st.write("Generating critical review...")
                result = generate_critical_review(content, chat_model)
                st.write(result)

            if st.button("Discussion Questions"):
                st.write("Generating discussion questions...")
                result = generate_discussion_questions(content, chat_model)
                st.write(result)

if __name__ == "__main__":
    main()