import active_liveness
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import cv2
import av
import numpy as np
from ocr.back_side_parse import get_info_back, get_string_similarity
import datetime
import time  # Import time module for delays
import os, sys
from active_liveness import headpose_liveness
from passive_liveness import predict_liveness
from face import dnn_face_detection, find_face_similarity
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image
import pytesseract
import numpy as np

# Global Variables
face_match = False
liveness = False

# Function Definitions
def find_target_list_regex(data, pattern):
    target_list = []
    for sublist in data:
        joined_str = ' '.join(sublist)
        match = re.findall(pattern, joined_str)
        if len(match) == 3:
            target_list.append(match)
    return target_list

def get_face(img):
    """Returns face image."""
    status = False
    face_box = dnn_face_detection.detect_face(img)
    if len(face_box) == 0:
        return status, "No Face Detected"
    elif len(face_box) >= 2:
        return status, "Multiple Faces Detected"
    elif len(face_box) == 1:
        box = face_box[0]
        x, y, w, h = box.astype('int')
        status = True
        face = img[y:h, x:w]
        return status, face

def levenshtein_similarity(str1, str2):
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    similarity = 1 - distance / max_len
    return similarity

def jaccard_similarity(str1, str2):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([str1, str2])
    similarity = cosine_similarity(vectors)[0, 1]
    return similarity

def image_to_array(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    lines_array = text.splitlines()
    return np.array(lines_array)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_frame = np.array([])

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        self.current_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format='bgr24')

# Streamlit UI Setup
st.title('Welcome to the AI Driven E-KYC portal')

# Create the progress bar
progress_bar = st.progress(0)

# Create the subtitle to show current step
progress_step = st.empty()

# Create the checkpoint labels for progress updates
checkpoints = [
    "Upload Images",
    "OCR Verification",
    "Face Match",
    "Liveness Detection"
]

citizenship_front = st.file_uploader('Upload clear image of the front side of your citizenship')
citizenship_back = st.file_uploader('Upload clear image of the back side of your citizenship')

# Proceed if both images are uploaded
if citizenship_front and citizenship_back:
    # Update progress bar after image upload
    progress_bar.progress(25)
    progress_step.subheader(checkpoints[1])  # Update subtitle to OCR Verification

    citizenship_front = np.asarray(bytearray(citizenship_front.read()), dtype=np.uint8)
    citizenship_front = cv2.imdecode(citizenship_front, 1)
    citizenship_back = np.asarray(bytearray(citizenship_back.read()), dtype=np.uint8)
    citizenship_back = cv2.imdecode(citizenship_back, 1)

    st.write('**Verification Step 1: OCR Verification Started**')
    try:
        stack_image = np.hstack([citizenship_front, cv2.resize(citizenship_back, (citizenship_front.shape[1], citizenship_front.shape[0]))])
        st.image(cv2.cvtColor(stack_image, cv2.COLOR_BGR2RGB), width=500)
    except:
        st.stop()

    # OCR extraction and user input
    info1 = get_info_back(citizenship_back)
    info = get_info_back(citizenship_front)
    target_pattern = r'\b\d{4}\b'
    target_list = find_target_list_regex(info, target_pattern)
    p_citizenship_number = ' '.join(target_list[0])

    flattened_list = [' '.join(sublist) for sublist in info1]
    address_index = flattened_list.index('Address')
    try:
        L_index = flattened_list.index(p_citizenship_number)
    except ValueError:
        L_index = None

    if L_index and L_index > address_index:
        text_address = ''.join(flattened_list[address_index + 1 : L_index])
    else:
        text_address = ""

    if len(p_citizenship_number) < 12:
        st.write('**Please upload clear images of documents and retry**')
        st.stop()

    full_name = st.text_input('Full Name', max_chars=30)
    citizenship_number = st.text_input('Citizenship number (without symbols)', max_chars=16)
    permanent_address = st.text_input('Permanent Address', max_chars=300)

    dob = st.date_input('Enter your DOB', min_value=datetime.date(1900,1,1))
    phone_no = st.text_input('Phone Number', max_chars=15)
    email_address = st.text_input('Email Address', max_chars=30)
    gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])

    if full_name and citizenship_number == p_citizenship_number:
        levenshtein_sim = levenshtein_similarity(text_address, permanent_address)
        jaccard_sim = jaccard_similarity(text_address, permanent_address)
        if levenshtein_sim > 0.5:
            st.subheader('OCR Verification Successful')
            # Update progress bar after OCR verification
            progress_bar.progress(50)
            progress_step.subheader(checkpoints[2])  # Update subtitle to Face Match

            # Display the button to start FaceMatch and Liveness Detection
            start_button = st.button('Start FaceMatch and Liveness Detection')

            if start_button:
                # Proceed to FaceMatch and Liveness detection
                st.write('**Proceeding to FaceMatch and Liveness Verification...**')
                # (Start face matching and liveness detection as described below)

                # Extract face from the document image
                status_card, face_card = get_face(citizenship_front)
                if not status_card:
                    st.subheader(f'{face_card} on uploaded document!')
                else:
                    st.image(cv2.cvtColor(face_card, cv2.COLOR_BGR2RGB), width=100)

                # Start webcam stream for face match and liveness
                ctx = webrtc_streamer(
                    client_settings=ClientSettings(
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                        media_stream_constraints={"video": True, "audio": False},
                    ),
                    video_processor_factory=VideoProcessor,
                    key="facematch-liveness",
                )

                if ctx.video_transformer:
                    while True:
                        selfie_image = ctx.video_transformer.current_frame
                        if selfie_image is not None:
                            status_selfie, selfie_face = get_face(selfie_image)
                            if not status_selfie:
                                st.subheader(f'{selfie_face} on captured image!')
                                break

                            stack_face = np.hstack([face_card, cv2.resize(selfie_face, (face_card.shape[1], selfie_face.shape[0]))])
                            st.image(cv2.cvtColor(stack_face, cv2.COLOR_BGR2RGB), width=200)

                            face_similarity = find_face_similarity.matching_prediction(face_card, selfie_face)
                            st.write(f'Face similarity: {face_similarity:.2f}**')

                            if face_similarity >= 0.5:
                                st.subheader('Facematch Verification Successful')
                                # Update progress bar after FaceMatch verification
                                progress_bar.progress(75)
                                progress_step.subheader(checkpoints[3])  # Update subtitle to Liveness Detection
                                st.write('**Starting Liveness Verification...**')
                                # (Include liveness detection logic here)

                                # Final progress update after Liveness detection
                                progress_bar.progress(100)
                            else:
                                st.markdown("## **OCR Verification Successful**")

                                break
        else:
            st.subheader('OCR Verification Failed, Enter Correct Information')
