import active_liveness 
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import cv2
import av
import numpy as np
from face import find_face_similarity
from ocr.back_side_parse import get_info_back, get_string_similarity
import datetime
import time  # Import time module for delays
import os, sys
from active_liveness import headpose_liveness
#from passive_liveness import predict_liveness
from face import dnn_face_detection
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image
import pytesseract
import numpy as np
import time
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import streamlit as st

#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
mp_face_detection = mp.solutions.face_detection


def detect_face_landmark(image, min_detection_confidence=0.7):
    with mp_face_detection.FaceDetection(min_detection_confidence) as face_detection:
        h, w = image.shape[:2]
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        keypoints = []
        if results.detections:
            for detection in results.detections:
                kp = list(detection.location_data.relative_keypoints)
                for i in range(6):
                    keypoints.append([int(kp[i].x * w), int(kp[i].y * h)])
        return keypoints


def get_head_pose(image):
    hpose = None
    keypoints = detect_face_landmark(image)
    if len(keypoints) != 0:
        rEye = tuple(keypoints[0])
        lEye = tuple(keypoints[1])
        nose = np.array(keypoints[2])
        mouth = np.array(keypoints[3])
        mid = (nose + mouth) / 2

        if lEye[0] != rEye[0]:
            slope = (lEye[1] - rEye[1]) / (lEye[0] - rEye[0])
            y_incpt = rEye[1] - (slope * rEye[0])
            y = slope * mid[0] + y_incpt
            k1 = distance.euclidean(rEye, (mid[0], int(y)))
            k2 = distance.euclidean((mid[0], int(y)), lEye)

            Rratio = 0 if k1 == 0 else k2 / k1
            Lratio = 0 if k2 == 0 else k1 / k2
            if Rratio <= 0.5:
                hpose = "right"
            elif Lratio <= 0.5:
                hpose = "left"
            else:
                hpose = "center"
    return hpose


# Global Variables
face_match = False
liveness = False
import cv2
import mediapipe as mp

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define indices for left and right eyes
LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 174, 175, 107, 108, 109, 110,
    323, 308, 286, 332, 327, 330, 291, 331, 306, 305, 304, 159, 158, 156, 130, 131,
    132, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 246, 161, 160, 143, 142, 141,
    37, 38, 39, 40, 41, 42
]
RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 372, 391, 367, 388, 368, 387, 386, 385, 384,
    398, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351, 350, 466, 389, 397, 366,
    365, 364, 263, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
    262, 38, 39, 40, 41, 42
]

def calculate_eye_aspect_ratio(eye_points, landmarks):
    """Calculates the eye aspect ratio (EAR) given eye landmarks."""
    # Convert landmarks to numpy arrays with their coordinates
    p1 = np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y])
    p2 = np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])
    p3 = np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y])
    p4 = np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
    
    # Calculate distances
    vertical_distance = np.linalg.norm(p1 - p2)
    horizontal_distance = np.linalg.norm(p3 - p4)
    
    # Avoid division by zero
    if vertical_distance == 0:
        return 0
    
    return horizontal_distance / (2.0 * vertical_distance)

def detect_liveness(frame):
    """Detects liveness based on eye blinking."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate EAR for both eyes
            left_eye_ear = calculate_eye_aspect_ratio(LEFT_EYE_INDICES, face_landmarks.landmark)
            right_eye_ear = calculate_eye_aspect_ratio(RIGHT_EYE_INDICES, face_landmarks.landmark)
            ear = (left_eye_ear + right_eye_ear) / 2.0
            
            # Threshold for detecting blinks (adjust as needed)
            blink_threshold = 0.2
            
            if ear < blink_threshold:
                return True  # Blinking detected, likely live
            else:
                return False  # No blinking detected, potentially spoofed
    
    return False  # No face detected


# Function Definitions

def stop_stream():
    # Raise an exception to signal stopping the stream
    raise Exception("Stop Stream")

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
        st.write('**Please enter Aadhar  no**')
        st.stop()

    full_name = st.text_input('Full Name', max_chars=30)
    citizenship_number = st.text_input('Aadhar number (without symbols)', max_chars=14)
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
            should_cont=True
            while ctx.video_transformer and not face_match:  # Add face_match condition
                selfie_image = ctx.video_transformer.current_frame
                if selfie_image is not None:
                    status_selfie, selfie_face = get_face(selfie_image)
                    if not status_selfie:
                        st.subheader(f'{selfie_face} on captured image!')
                        break

                    # Resize selfie_face to match face_card's height
                    resized_selfie_face = cv2.resize(selfie_face, (face_card.shape[1], face_card.shape[0]))

                    # Stack faces horizontally
                    stack_face = np.hstack([face_card, resized_selfie_face])
                    st.image(cv2.cvtColor(stack_face, cv2.COLOR_BGR2RGB), width=200)

                    face_similarity = find_face_similarity.matching_prediction(face_card, selfie_face)
                    
                    if face_similarity >= 0.5:
                        st.write(f'Face similarity: {face_similarity*100:.2f}%')
                        st.success("FaceMatch Verification Successful!")
                        progress_bar.progress(75)
                        progress_step.subheader(checkpoints[3])
                        st.subheader('liveliness detection')
                        face_match = True
                        break
                    break
                break
                    
            if face_match:
                st.write("### Liveness Detection Instructions:")
                st.write("1. Make sure you're in a well-lit area")
                st.write("2. Position your face clearly in the camera")
                st.write("3. When ready, click 'Start Liveness Detection'")
                st.write("4. Blink naturally 3-4 times over the next 10 seconds")
                st.write("5. Keep your head steady and face the camera")
                
                liveliness_button = st.button("Start Liveness Detection")
                
                if liveliness_button:
                    st.write("**Starting liveness detection...**")
                    st.write("Simply blink naturally - no need to exaggerate your blinks")
                    
                    # Start countdown
                    progress_text = st.empty()
                    progress_text.write("Time remaining: 10 seconds")
                    
                    blink_counter = 0
                    start_time = time.time()
                    last_blink_time = time.time()
                    was_blinking = False
                    
                    while ctx.video_transformer and time.time() - start_time < 10:
                        # Update countdown
                        remaining_time = 10 - int(time.time() - start_time)
                        progress_text.write(f"Time remaining: {remaining_time} seconds")
                        
                        frame = ctx.video_transformer.current_frame
                        if frame is not None:
                            is_live = detect_liveness(frame)
                            
                            if is_live and not was_blinking and time.time() - last_blink_time > 0.5:
                                blink_counter += 1
                                last_blink_time = time.time()
                                st.write(f"✅ Blink {blink_counter}/3 detected!")
                            
                            was_blinking = is_live
                            
                            # Show feedback
                            status_text = "Blink detected!" if is_live else "Keep blinking naturally..."
                            frame_with_text = frame.copy()
                            cv2.putText(frame_with_text, status_text, (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                    (0, 255, 0) if is_live else (0, 0, 255), 2)
                            st.image(cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB), 
                                    channels="RGB", use_column_width=True)
                            
                            if blink_counter >= 3:
                                st.success("✨ Liveness Detection Successful! Thank you!")
                                progress_bar.progress(100)
                                liveness = True
                                break
                            
                            time.sleep(0.1)
                    
                    if not liveness:
                        st.error("Liveness Detection Failed. Please try again.")
                        st.write("Tips for success:")
                        st.write("- Make sure your face is well-lit")
                        st.write("- Keep your head steady")
                        st.write("- Blink naturally - don't force it")