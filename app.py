import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import torch
from pdf2image import convert_from_path
import pandas as pd
import io
import os
from tempfile import NamedTemporaryFile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import qrcode

# Load model and tokenizer
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

# Function to classify entities based on custom rules
def classify_entity(word, label):
    if "ADDR" in label:
        return "Address"
    elif "DATE" in label or any(char.isdigit() for char in word):
        return "Date"
    elif "TOTAL" in label:
        return "Total"
    elif "PRICE" in label or word.startswith('$') or word.replace('.', '', 1).isdigit():
        return "Price"
    elif "NAME" in label or word.istitle():
        return "Name"
    elif "ITEM" in label:
        return "Item"
    else:
        return "Other"

# Function to extract and segment entities
def extract_and_segment_entities(image):
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = data['text']

        if not words or all(word.strip() == '' for word in words):
            raise ValueError("No text detected, switching to fallback processing.")

        encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=2).tolist()[0]

        # Initialize segments dictionary
        segments = {
            "Address": [],
            "Date": [],
            "Total": [],
            "Item": [],
            "Price": [],
            "Name": [],
            "Other": []
        }

        for i, word in enumerate(words):
            label = model.config.id2label[predictions[i]]
            if word.strip():  # Ensure it's not an empty word
                category = classify_entity(word, label)
                entity = {
                    "word": word,
                    "label": category,
                    "box": (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                }
                segments[category].append(entity)

        return segments, words

    except Exception as e:
        st.warning(f"Original OCR process failed with error: {e}. Trying fallback processing...")
        try:
            # Fallback processing code
            image = process_with_fallback(image)  # Process the image using fallback method
            
            # Display the fallback processed document
            st.image(image, caption="Fallback Processed Document", use_column_width=True)
            
            # Re-run the original process on the newly processed image
            return extract_and_segment_entities(image)
        except Exception as fallback_error:
            st.error(f"Both the original and fallback OCR processes failed. Error: {fallback_error}")
            return {}, []  # Return empty results to indicate failure

# Fallback processing function
def process_with_fallback(image):
    # Placeholder for actual fallback processing logic, e.g., image pre-processing, enhancement, etc.
    st.write("Applying fallback processing...")
    image = image.convert("L")  # Convert to grayscale as an example
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Simple thresholding
    return image

# Function to draw boxes on image with color coding and transparency
def draw_segmented_boxes(image, segments, fill=False):
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    color_map = {
        "Address": (0, 0, 255, 100),  # Blue
        "Date": (0, 200, 0, 100),  # Deep Green (Adjusted)
        "Total": (255, 0, 0, 100),  # Red
        "Item": (255, 165, 0, 100),  # Orange
        "Price": (0, 255, 255, 100),  # Cyan
        "Name": (255, 0, 255, 100),  # Magenta
        "Other": (128, 0, 128, 100)  # Purple
    }

    for segment, entities in segments.items():
        color = color_map.get(segment, (0, 0, 0, 100))  # Default to black with some transparency
        for entity in entities:
            box = (entity['box'][0], entity['box'][1], entity['box'][0] + entity['box'][2],
                   entity['box'][1] + entity['box'][3])
            if fill:
                draw.rectangle(box, fill=color, outline=color[:-1] + (255,),
                               width=2)  # Fill with transparency, solid border
                text_position = (entity['box'][0] + 2, entity['box'][1] + 2)
                draw.text(text_position, entity['word'], fill="black", font=font)
            else:
                draw.rectangle(box, outline=color[:-1] + (255,), width=2)  # Only draw border
                # Draw the original text above the bounding box
                text_bbox = draw.textbbox((0, 0), entity['word'], font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_position = (entity['box'][0], entity['box'][1] - text_height - 2)
                draw.text(text_position, entity['word'], fill=color[:-1] + (255,), font=font)

    return image

# Function to load PDF as images
def load_images_from_pdf(uploaded_pdf):
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_pdf.read())
            temp_pdf_path = temp_pdf.name
        
        images = convert_from_path(temp_pdf_path)
        os.remove(temp_pdf_path)  # Clean up the temporary file after conversion
        return images
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")
        return []

# Function to convert text to PDF using ReportLab
def convert_text_to_pdf(text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 40, "Extracted OCR Text:")
    text_object = c.beginText(30, height - 60)
    text_object.setFont("Helvetica", 12)
    text_object.textLines(text)
    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Video processing class for camera input
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.image = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.image = img
        return img

# Streamlit app
st.title("Document Information Extraction and Segmentation with Camera Input")

# Option to upload document or use camera
option = st.selectbox(
    "Choose how to provide the document:",
    ("Upload Document", "Scan Document with Internal Camera", "Scan Document with External Camera (Phone)")
)

# Initialize variables
image = None

if option == "Upload Document":
    uploaded_file = st.file_uploader("Upload a document", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            images = load_images_from_pdf(uploaded_file)
            if images:
                image = images[0]  # Process only the first page for simplicity
        else:
            image = Image.open(uploaded_file)

elif option == "Scan Document with Internal Camera":
    st.write("Use the internal camera to scan the document.")
    webrtc_ctx = webrtc_streamer(
        key="internal-camera",
        video_transformer_factory=VideoTransformer,
        client_settings={"video": {"width": 1280, "height": 720}}
    )

    if webrtc_ctx.video_transformer:
        if st.button("Capture Document"):
            img_array = webrtc_ctx.video_transformer.image
            if img_array is not None:
                image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

elif option == "Scan Document with External Camera (Phone)":
    st.write("Scan the QR code below with your phone to open the camera:")
    
    # Generate QR code
    public_url = "https://docuscanapp.streamlit.app/"  # Replace with your public Streamlit app URL
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(public_url)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")

    # Convert PIL Image to bytes-like object for Streamlit
    buf = io.BytesIO()
    img_qr.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.image(byte_im, caption="Scan this QR code with your phone to use its camera.")

    st.write("Once you scan the QR code, use your phone's browser to capture the document image.")

    if st.button("Capture Document from Phone"):
        st.write("Use your phone to scan the document and upload it here.")

if image is not None:
    segments, extracted_words = extract_and_segment_entities(image)
    
    if not segments:
        st.error("No text detected in the document. Please try scanning again.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Original Document")
            st.image(image, caption='Uploaded/Scanned Document', use_column_width=True)

        with col2:
            st.header("Segmented Document")
            filled_annotated_image = image.copy()
            filled_annotated_image = draw_segmented_boxes(filled_annotated_image, segments, fill=True)
            st.image(filled_annotated_image, caption="Segmented Document", use_column_width=True)

        with col3:
            st.header("Segmented Details")
            annotated_image = image.copy()
            annotated_image = draw_segmented_boxes(annotated_image, segments, fill=False)
            st.image(annotated_image, caption="Segmented Details", use_column_width=True)

        st.header("Extracted OCR Text")

        # Displaying the extracted text with a scrollable container
        formatted_text = "\n".join(extracted_words)

        st.markdown(
            f"""
            <div style='max-height: 200px; overflow-y: scroll; padding: 5px; background-color: #f9f9f9; border-radius: 5px;'>
                <pre style='font-size: 16px; font-family: Arial, sans-serif; white-space: pre-wrap;'>{formatted_text}</pre>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Dropdown for download options
        st.subheader("Download Extracted OCR Text")
        download_option = st.selectbox(
            "Choose download format",
            ("Text", "PDF", "Excel")
        )

        if download_option == "Text":
            # Convert text to bytes for download
            text_bytes = formatted_text.encode('utf-8')
            st.download_button(
                label="Download as Text",
                data=text_bytes,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

        elif download_option == "PDF":
            # Convert text to PDF using ReportLab
            pdf_buffer = convert_text_to_pdf(formatted_text)
            st.download_button(
                label="Download as PDF",
                data=pdf_buffer,
                file_name="extracted_text.pdf",
                mime="application/pdf"
            )

        elif download_option == "Excel":
            # Convert text to Excel using openpyxl engine
            df = pd.DataFrame({"Extracted Text": extracted_words})
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            excel_buffer.seek(0)
            st.download_button(
                label="Download as Excel",
                data=excel_buffer,
                file_name="extracted_text.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
