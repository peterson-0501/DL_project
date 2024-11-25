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


# Initialize the document-question-answering pipeline
# pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")


# def doc_que_ans(uploaded_document, question):
#     """
#     Function to process a document and return the answer to the question.
#
#     Args:
#     - uploaded_document: The document uploaded by the user.
#     - question: The question asked by the user.
#
#     Returns:
#     - answer: The answer extracted from the document.
#     """
#
#     # Extract text from the uploaded document
#     if uploaded_document.type == "application/pdf":
#         # Extract text from PDF
#         pdf_reader = PyPDF2.PdfReader(uploaded_document)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#
#     elif uploaded_document.type == "text/plain":
#         # Extract text from plain text file
#         text = uploaded_document.read().decode("utf-8")
#
#     elif uploaded_document.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#         # Extract text from DOCX file
#         doc = Document(uploaded_document)
#         text = "\n".join([para.text for para in doc.paragraphs])
#
#     # Use the pipeline to get the answer to the question
#     result = pipe(question=question, context=text)
#
#     return result['answer']


# ##################################stremlit implimentiation#############################
# import streamlit as st
# # Set page configuration
# st.set_page_config(
#     page_title="AI Tools Hub",
#     page_icon=":robot:",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # Sidebar for navigation
# st.sidebar.title("AI Tools Hub")
# st.sidebar.subheader("Choose a Category:")
# category = st.sidebar.radio("Category", ["Home", "Computer Vision", "NLP"])
#
# if category == "Home":
#     st.title("Welcome to AI Tools Hub")
#     st.markdown("""
#     Explore state-of-the-art AI tools in **Computer Vision** and **Natural Language Processing (NLP)**.
#     Select a category from the sidebar to get started:
#
#     - **Computer Vision**:
#         - Image Classification
#         - Text-to-Image
#         - Image-to-Text
#
#     - **NLP**:
#         - Text Classification
#         - Summarization
#         - Text Generation
#         - Question Answering
#     """)
#     st.balloons()
#
# elif category == "Computer Vision":
#     st.title(":camera: Computer Vision Tools")
#     task = st.sidebar.radio(
#         "Select a task:",
#         ["Image Classification", "Text-to-Image", "Image-to-Text"]
#     )
#
#     if task == "Image Classification":
#         st.header("Image Classification")
#         st.write("Upload an image to classify objects.")
#
#         # File uploader for image input
#         uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#
#         if uploaded_file is not None:
#             # Display the uploaded image
#             st.image(uploaded_file, caption="Uploaded Image")# use_column_width=True)
#
#             #st.write("Classifying image... Please wait.")
#
#             # Call the image classification function
#             with st.spinner("Processing..."):
#                 result = imageclass(uploaded_file)
#
#             # Display the result
#             if "error" in result:
#                 st.error(f"Error: {result['error']}")
#             else:
#                 st.write("Prediction Results:")
#                 for prediction in result:
#                     st.write(f"**{prediction['label']}**: {prediction['score']:.2%}")
#
#
#     # elif task == "Image Generation":
#     #     st.header("Image Generation")
#     #     st.write("Generate images from random seeds or specific prompts.")
#     #     prompt = st.text_input("Enter a prompt for image generation:")
#     #     if prompt:
#     #         st.write(f"Generating an image for: **{prompt}**")
#     #         # Call your image generation model here
#     #         st.image("https://via.placeholder.com/400", caption="Generated Image (placeholder)")
#
#     elif task == "Text-to-Image":
#         st.header("Text-to-Image Generation")
#         st.write("Transform a text description into an image.")
#         description = st.text_area("Enter a detailed description for the image:")
#         if description:
#             #st.write(f"Generating an image for: **{description}**")
#             # Call the image classification function
#             with st.spinner("Processing..."):
#                 generated_image = text_to_image(description)
#             if generated_image:
#                 st.image(generated_image, caption=f"Generated Image for: {description}",use_column_width=True)
#             else:
#                 st.error("Failed to generate image. Please try again.")
#
#
#     elif task == "Image-to-Text":
#         st.header("Image-to-Text (Captioning)")
#         st.write("Upload an image to generate a caption.")
#         uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#         if uploaded_file is not None:
#             st.image(uploaded_file, caption="Uploaded Image", use_column_width=200)
#             with st.spinner("Processing..."):
#                 generated_caption = image_to_text(uploaded_file)
#                 # Display the dynamically generated caption
#                 st.write(f"Generated text: **{generated_caption[0]["generated_text"]}**")
#
# elif category == "NLP":
#     st.title(":pencil: Natural Language Processing Tools")
#     task = st.sidebar.radio(
#         "Select a task:",
#         ["Text Classification", "Summarization","Text Generation","Question Answering"]
#     )
#
#     if task == "Text Classification":
#         st.header("Text Classification")
#         st.write("Classify text into categories.")
#         text = st.text_area("Enter text for classification:")
#         if text:
#             with st.spinner("Processing..."):
#                 text_class = text_classification(text)
#
#             # Convert to DataFrame
#             df = pd.DataFrame(text_class[0])
#
#             # Display in Streamlit
#             st.title("Text Classification")
#             st.write("Here are the classification scores:")
#
#             # Display table
#             #st.table(df)  # For a static table
#             st.dataframe(df)  # For an interactive table
#
#
#
#
#     elif task == "Summarization":
#         st.header("Text Summarization")
#         st.write("Summarize long passages into concise text.")
#         text = st.text_area("Enter text to summarize:")
#         if text:
#             with st.spinner("Processing..."):
#                 text_sum = text_summary(text)
#             st.write(f"Summary : **{text_sum[0]["summary_text"]}**")
#
#     elif task == "Text Generation":
#         st.header("Text Generation")
#         st.write("Generate text based on your input prompt.")
#         user_input = st.text_area("Enter your prompt:", placeholder="Type something...")
#         if st.button("Generate Text"):
#             if user_input.strip():
#                 with st.spinner("Processing..."):
#                     generated_text = text_gen(user_input)
#                     st.write(f"Generated text: **{generated_text[0]["generated_text"]}**")
#             else:
#                 st.warning("Please enter a valid prompt.")
#
#     elif task == "Question Answering":
#         st.header("Question Answering")
#         st.write("Ask a question based on the provided context.")
#         context = st.text_area("Enter Context:", placeholder="Type somethign...")
#         question = st.text_input("Enter Question:", placeholder="Type somethign...")
#
#         if st.button("Get Answer"):
#             if question.strip() and context.strip():
#                 with st.spinner("Processing..."):
#                     get_answer = question_ans(context,question)
#                     st.write(f"Answer: **{get_answer}**")
#
#             else:
#                 st.warning("Please enter a valid Question/Context.")
#
#
#
# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("Developed with ❤️ using Streamlit")

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

