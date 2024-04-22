from rmn import RMN
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from collections import Counter
from play_song import play_random_song
import random
import threading


model = RMN()

def process_image(image):
    image = np.array(image)
    emotion = model.detect_emotion_for_single_frame(image)
    return emotion

def capture_face():
    capture_duration = 5  # Capture images for 5 seconds
    cap = cv2.VideoCapture(0)  # Open the camera (0 for default camera)

    images = []
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb_frame))
        
        # Check if capture duration is reached
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time >= capture_duration:
            break

    cap.release()  # Release the camera
    
    emotions = []
    for image in images:
        emotions.append(process_image(image))
    emotion_counts = Counter(emotions)
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    
    return most_common_emotion

def play_and_monitor_emotion():
    global is_playing, curr_emo, thread

    curr_emo = capture_face()
    play_random_song(curr_emo)

    # Wait for the song to finish playing
    is_playing = True
    while is_playing:
        # Check if the thread is alive (song is still playing)
        if not thread.is_alive():
            is_playing = False

def main():
    global is_playing, curr_emo, thread

    st.title("Face Capture and Model Output")

    if not is_playing:
        if st.button("Play"):
            thread = threading.Thread(target=play_and_monitor_emotion)
            thread.start()

    if is_playing:
        st.write(f"Current Emotion: {curr_emo}")
        if st.button("Stop"):
            is_playing = False
            thread.join()  # Wait for the song thread to complete
            st.success("Song Stopped!")

if __name__ == "__main__":
    is_playing = False
    curr_emo = None
    thread = None
    main()
