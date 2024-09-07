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
    st.title("Scientific Article Analysis App")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [SystemMessage(content="You are an AI assistant answering questions about a scientific article.")]

    uploaded_file = st.file_uploader("Upload your scientific article (PDF)", type="pdf")

    if uploaded_file is not None:
        if 'content' not in st.session_state:
            st.session_state.content = extract_text_from_pdf(uploaded_file)
            st.session_state.images = extract_images_from_pdf(uploaded_file)
            logger.info("PDF content extracted and stored in session state")

        chat_model = create_chat_model()

        st.write("Select a feature to analyze the paper:")

        # Create two rows of buttons
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        with col1:
            background_context = st.button("Background Context")
        with col2:
            paper_summary = st.button("Paper Summary")
        with col3:
            question_answering = st.button("Question Answering")
        with col4:
            figure_analysis = st.button("Figure Analysis")
        with col5:
            critical_review = st.button("Critical Review")
        with col6:
            discussion_questions = st.button("Discussion Questions")

        # Create a container for the output
        output_container = st.container()

        with output_container:
            if background_context:
                st.write("Generating background context...")
                result = generate_background_context(st.session_state.content, chat_model)
                st.markdown(result)

            elif paper_summary:
                st.write("Generating paper summary...")
                result = generate_paper_summary(st.session_state.content, chat_model)
                st.write(result)

            elif question_answering:
                st.write("Question Answering Mode Activated")
                
                # Display chat history
                for message in st.session_state.chat_history[1:]:  # Skip the system message
                    st.write(f"{'You' if isinstance(message, HumanMessage) else 'AI'}: {message.content}")

                # Get user input
                user_input = st.text_input("Your question:")
                submit_button = st.button("Submit Question")

                if submit_button:
                    if user_input:
                        logger.info(f"Question submitted: {user_input}")
                        st.session_state.chat_history.append(HumanMessage(content=user_input))
                        try:
                            response = chat_model(st.session_state.chat_history + [HumanMessage(content=f"Article content: {st.session_state.content}")])
                            logger.info("Response generated successfully")
                            ai_message = AIMessage(content=response.content)
                            st.session_state.chat_history.append(ai_message)
                            st.write(f"AI: {ai_message.content}")
                        except Exception as e:
                            logger.error(f"Error generating response: {str(e)}")
                            st.error(f"An error occurred while generating the response: {str(e)}")
                    else:
                        st.warning("Please enter a question before submitting.")

            elif figure_analysis:
                st.write("Analyzing figures...")
                result, figures = analyze_figures(st.session_state.content, st.session_state.images, chat_model)
                st.write(result)
                for i, img in enumerate(figures):
                    st.image(img, caption=f"Figure {i+1}", use_column_width=True)

            elif critical_review:
                st.write("Generating critical review...")
                result = generate_critical_review(st.session_state.content, chat_model)
                st.write(result)

            elif discussion_questions:
                st.write("Generating discussion questions...")
                result = generate_discussion_questions(st.session_state.content, chat_model)
                st.write(result)

if __name__ == "__main__":
    main()