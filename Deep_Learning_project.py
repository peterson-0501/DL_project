import tempfile
import requests
import os
import io
from PIL import Image
import pandas as pd
# from transformers import pipeline
# import PyPDF2
# from docx import Document


def imageclass(uploaded_file):
    import requests

    # Hugging Face API details
    API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
    headers = {"Authorization": "Bearer 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"}

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        temp_filename = temp.name

    # Pass temp file to query
    output = query(temp_filename)

    # Clean up temp file
    os.remove(temp_filename)

    return output

def text_to_image(prompt):


    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    image_bytes = query({
        "inputs": prompt
    })

    output = Image.open(io.BytesIO(image_bytes))
    return output


def image_to_text(uploaded_file):
    import requests

    # Hugging Face API details
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": "Bearer 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"}


    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        temp_filename = temp.name

    # Pass temp file to query
    output = query(temp_filename)

    # Clean up temp file
    os.remove(temp_filename)

    return output


def text_classification(text):
    API_URL = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text
    })

    return output


def text_summary(text):
    import requests

    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text})

    return output


def text_gen(text):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text
    })

    return output


def question_ans(Context,Question):
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": {
            "question": Question,
            "context": Context}
    })
    return output



import streamlit as st
from streamlit.components.v1 import html

# Set page configuration
st.set_page_config(
    page_title="AI Tools Hub",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #4CAF50;
            margin-bottom: 1rem;
        }
        .description {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .st-sidebar .sidebar-content {
            background-color: #F8F9FA;
        }
        .st-sidebar h1 {
            color: #333;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            font-size: 0.9rem;
            color: #777;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("AI Tools Hub")
st.sidebar.subheader("Choose a Category:")
category = st.sidebar.radio("Category", ["Home", "Computer Vision", "NLP"])

if category == "Home":
    st.markdown('<div class="main-title">Welcome to AI Tools Hub</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="description">
            Explore state-of-the-art AI tools in <strong>Computer Vision</strong> and <strong>Natural Language Processing (NLP)</strong>.  
            Select a category from the sidebar to get started:
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        - **Computer Vision**:  
            - Image Classification  
            - Text-to-Image  
            - Image-to-Text 
        - **NLP**:  
            - Text Classification  
            - Summarization  
            - Text Generation  
            - Question Answering  
    """)

    st.balloons()

elif category == "Computer Vision":
    st.title(":camera: Computer Vision Tools")
    task = st.sidebar.radio(
        "Select a task:",
        ["Image Classification", "Text-to-Image", "Image-to-Text"]
    )

    if task == "Image Classification":
        st.header("Image Classification")
        st.write("Upload an image to classify objects.")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image")
            with st.spinner("Processing..."):
                result = imageclass(uploaded_file)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.write("Prediction Results:")
                for prediction in result:
                    st.write(f"**{prediction['label']}**: {prediction['score']:.2%}")

    elif task == "Text-to-Image":
        st.header("Text-to-Image Generation")
        st.write("Transform a text description into an image.")
        description = st.text_area("Enter a detailed description for the image:")
        if description:
            with st.spinner("Processing..."):
                generated_image = text_to_image(description)
            if generated_image:
                st.image(generated_image, caption=f"Generated Image for: {description}")
            else:
                st.error("Failed to generate image. Please try again.")

    elif task == "Image-to-Text":
        st.header("Image-to-Text (Captioning)")
        st.write("Upload an image to generate a caption.")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image")
            with st.spinner("Processing..."):
                generated_caption = image_to_text(uploaded_file)
                st.write(f"Generated text: **{generated_caption[0]['generated_text']}**")


elif category == "NLP":
    st.title(":pencil: Natural Language Processing Tools")
    task = st.sidebar.radio(
        "Select a task:",
        ["Text Classification", "Summarization", "Text Generation", "Question Answering"]
    )

    if task == "Text Classification":
        st.header("Text Classification")
        text = st.text_area("Enter text for classification:")
        if text:
            with st.spinner("Processing..."):
                text_class = text_classification(text)
            st.write("Classification Results:")
            st.dataframe(pd.DataFrame(text_class[0]))

    elif task == "Summarization":
        st.header("Text Summarization")
        text = st.text_area("Enter text to summarize:")
        if text:
            with st.spinner("Processing..."):
                text_sum = text_summary(text)
            st.write(f"Summary: **{text_sum[0]['summary_text']}**")

    elif task == "Text Generation":
        st.header("Text Generation")
        user_input = st.text_area("Enter your prompt:")
        if st.button("Generate Text"):
            with st.spinner("Processing..."):
                generated_text = text_gen(user_input)
            st.write(f"Generated text: **{generated_text[0]['generated_text']}**")

    elif task == "Question Answering":
        st.header("Question Answering")
        context = st.text_area("Enter Context:")
        question = st.text_input("Enter Question:")
        if st.button("Get Answer"):
            with st.spinner("Processing..."):
                get_answer = question_ans(context, question)
            st.write(f"Answer: **{get_answer}**")

# elif category == "Multimodal":
#     st.title("Multimodal Tools")
#     task = st.sidebar.radio(
#         "Select a task:",
#         ["Document Question Answering"]
#     )
#     if task == "Document Question Answering":
#         uploaded_document = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])
#         if uploaded_document:
#             question = st.text_input("Ask a Question")
#             if question:
#                 with st.spinner("Processing..."):
#                     # Call the doc_que_ans function to get the answer
#                     answer = doc_que_ans(uploaded_document, question)
#                     st.write("Answer:", answer)


# Footer with developer contact
st.markdown("""
    <div class="footer">
        <hr>
        <p>Developed with ❤️ by <a href="mailto:annunaypandey@gmail.com">Annunay Pandey</a></p>
        <p>Contact: <a href="https://www.linkedin.com/in/annunay-pandey/" target="_blank">LinkedIn</a> | <a href="https://github.com/peterson-0501" target="_blank">GitHub</a></p>
    </div>
""", unsafe_allow_html=True)

