import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2 as pdf
import docx2txt 
import base64
from io import BytesIO
import os

# Set page config - moved to the top to avoid config issues
st.set_page_config(page_title="Word Cloud Generator", layout="wide")

# Title
st.title("📝 Word Cloud Generator")
st.write("Upload text files, PDFs, or Word documents to generate a word cloud")

# File uploader
uploaded_files = st.file_uploader(
    "Upload your files (TXT, PDF, DOCX)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

# Word cloud parameters
st.sidebar.header("Word Cloud Settings")
width = st.sidebar.slider("Width", 400, 1200, 800)
height = st.sidebar.slider("Height", 200, 800, 400)
background_color = st.sidebar.color_picker("Background Color", "#FFFFFF")
max_words = st.sidebar.slider("Max Words", 50, 500, 200)
colormap = st.sidebar.selectbox(
    "Color Map",
    ["viridis", "plasma", "inferno", "magma", "cividis", "cool", "hot", "spring", "summer", "autumn", "winter"]
)
contour_width = st.sidebar.slider("Contour Width", 0, 10, 0)
contour_color = st.sidebar.color_picker("Contour Color", "#000000")

def extract_text_from_file(file):
    """Extract text from different file types"""
    try:
        if file.type == "application/pdf":
            reader = pdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif file.type == "text/plain":
            return str(file.read(), "utf-8")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(file)
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return ""
    return ""

def generate_word_cloud(text):
    """Generate and display word cloud"""
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            colormap=colormap,
            contour_width=contour_width,
            contour_color=contour_color
        ).generate(text)
        
        # Create figure with tight layout to prevent cutting off
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        
        # Use st.pyplot() with clear_figure=True to prevent memory leaks
        st.pyplot(plt, clear_figure=True)
        
        # Add download button
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        
        st.download_button(
            label="Download Word Cloud",
            data=img_buffer,
            file_name="wordcloud.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text:
            all_text += text + " "
    
    if all_text.strip():
        st.subheader("Generated Word Cloud")
        generate_word_cloud(all_text)
        
        # Show text statistics
        word_count = len(all_text.split())
        char_count = len(all_text)
        unique_words = len(set(all_text.split()))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Words", word_count)
        col2.metric("Unique Words", unique_words)
        col3.metric("Characters", char_count)
        # Show raw text (optional)
        with st.expander("Show extracted text"):
            st.text(all_text[:5000] + "..." if len(all_text) > 5000 else all_text)
    else:
        st.warning("No text could be extracted from the uploaded files.")
else:
    st.info("Please upload files to generate a word cloud.")

# Add some styling
st.markdown("""
<style>
    .stDownloadButton button {
        width: 100%;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)