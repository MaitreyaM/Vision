import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv

load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent - Media Summarizer",
    page_icon="🎥🖼️",
    layout="wide"
)

st.title("Phidata Media AI Summarizer Agent 🎥🖼️")
st.header("Powered by Gemini 2.0 Flash Exp")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Media AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )


# Initialize the agent
multimodal_Agent = initialize_agent()

# Tab layout for videos, images, and text
tabs = st.tabs(["🎥 Video Analysis", "🖼️ Image Analysis", "📝 Text Summarization"])

# Video Analysis Tab
with tabs[0]:
    video_file = st.file_uploader(
        "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
    )

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)

        user_query = st.text_area(
            "What insights are you seeking from the video?",
            placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
            help="Provide specific questions or insights you want from the video."
        )

        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        # Upload and process video file
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        # Prompt generation for analysis
                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query using video insights and supplementary web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """
                        )

                        # AI agent processing
                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                    # Display the result
                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    # Clean up temporary video file
                    Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Upload a video file to begin analysis.")

# Image Analysis Tab
with tabs[1]:
    image_file = st.file_uploader(
        "Upload an image file", type=['jpg', 'jpeg', 'png'], help="Upload an image for AI analysis"
    )

    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
            temp_image.write(image_file.read())
            image_path = temp_image.name

        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        user_query = st.text_area(
            "What insights are you seeking from the image?",
            placeholder="Ask anything about the image content. The AI agent will analyze and provide insights.",
            help="Provide specific questions or insights you want from the image."
        )

        if st.button("🔍 Analyze Image", key="analyze_image_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the image.")
            else:
                try:
                    with st.spinner("Processing image and gathering insights..."):
                        # Upload and process image file
                        processed_image = upload_file(image_path)
                        while processed_image.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_image = get_file(processed_image.name)

                        # Prompt generation for analysis
                        analysis_prompt = (
                            f"""
                            Analyze the uploaded image for content and context.
                            Perform OCR if text is present or describe the image otherwise.
                            Respond to the following query using image insights:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """
                        )

                        # AI agent processing
                        response = multimodal_Agent.run(analysis_prompt, images=[processed_image])

                    # Display the result
                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    # Clean up temporary image file
                    Path(image_path).unlink(missing_ok=True)
    else:
        st.info("Upload an image file to begin analysis.")

# Text Summarization Tab
with tabs[2]:
    st.subheader("Text Summarization")
    
    # Text Input Area
    text_input = st.text_area(
        "Enter your text for summarization:",
        placeholder="Type or paste text here...",
        help="Provide text directly for summarization."
    )
    
    # Text File Upload
    text_file = st.file_uploader(
        "Or upload a text file:", type=['txt'], help="Upload a text file containing the content to summarize."
    )

    if text_file:
        # Read content from uploaded text file
        text_content = text_file.read().decode("utf-8")
    else:
        # Use manually entered text
        text_content = text_input

    # Prompt Input Area
    summary_prompt_input = st.text_input(
        "Customize your summarization prompt:",
        placeholder="For example: Summarize with key highlights, or provide bullet points.",
        help="Specify additional instructions for how the text should be summarized."
    )

    # Summarization Button
    if st.button("🔍 Summarize Text", key="summarize_text_button"):
        if not text_content.strip():
            st.warning("Please provide some text or upload a file to summarize.")
        else:
            try:
                with st.spinner("Summarizing text..."):
                    # Generate summarization prompt
                    summary_prompt = (
                        f"""
                        Summarize the following text in a concise, clear, and user-friendly manner.
                        {summary_prompt_input if summary_prompt_input.strip() else ""}
                        
                        Text to summarize:
                        {text_content}
                        """
                    )

                    # AI agent processing
                    response = multimodal_Agent.run(summary_prompt)

                # Display the result
                st.subheader("Summary Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during summarization: {error}")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)