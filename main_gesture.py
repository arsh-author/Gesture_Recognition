import tkinter as tk
from tkinter import messagebox
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import threading

# Global variables
bg = None
model = None
camera = None
stop_event = threading.Event()

# ROI coordinates
roi_top, roi_right, roi_bottom, roi_left = 10, 350, 225, 590

def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def _load_weights():
    global model
    try:
        model = load_model("hand_gesture_recog_model3.h5")
        print(model.summary())
    except Exception as e:
        print(f"Error loading model: {e}")

def getPredictedClass():
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, [100, 120])
    gray_image = gray_image.reshape(1, 100, 120, 1)
    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)
    classes = ["Blank", "OK", "Thumbs Up", "Thumbs Down", "Punch", "High Five"]
    return classes[predicted_class]

def update_status(message):
    status_label.config(text=message)

def update_frame():
    global camera, stop_event
    if stop_event.is_set():
        return

    grabbed, frame = camera.read()
    if not grabbed:
        return

    frame = imutils.resize(frame, width=800)
    frame = cv2.flip(frame, 1)
    (height, width) = frame.shape[:2]

    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if not hasattr(update_frame, "num_frames"):
        update_frame.num_frames = 0

    if update_frame.num_frames < 30:
        run_avg(gray, 0.5)
        if update_frame.num_frames == 1:
            update_status("Calibration in progress...")
        elif update_frame.num_frames == 29:
            update_status("Calibration successful.")
    else:
        hand = segment(gray)
        if hand is not None:
            (thresholded, segmented) = hand
            cv2.drawContours(frame, [segmented + (roi_right, roi_top)], -1, (0, 0, 255))
            if update_frame.num_frames % (int(camera.get(cv2.CAP_PROP_FPS)) // 6) == 0:
                cv2.imwrite('Temp.png', thresholded)
                predictedClass = getPredictedClass()
                cv2.putText(frame, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Thresholded", thresholded)

    # Draw the ROI rectangle on the frame
    cv2.rectangle(frame, (roi_right, roi_top), (roi_left, roi_bottom), (0, 255, 0), 2)

    update_frame.num_frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    
    video_label.imgtk = img
    video_label.config(image=img)
    video_label.after(10, update_frame)

def start_recognition():
    global camera, stop_event
    stop_event.clear()
    if camera is None:
        camera = cv2.VideoCapture(0)
    update_status("Starting recognition...")
    update_frame()

def stop_recognition():
    global stop_event, camera
    stop_event.set()
    if camera:
        camera.release()
        camera = None
    cv2.destroyAllWindows()
    video_label.config(image=None)  # Clear the video feed from the label
    update_status("Recognition stopped.")

def show_main_app():
    welcome_frame.pack_forget()  # Hide the welcome frame
    main_frame.pack(fill=tk.BOTH, expand=True)  # Show the main frame

def show_welcome_screen():
    main_frame.pack_forget()  # Hide the main frame
    welcome_frame.pack(fill=tk.BOTH, expand=True)  # Show the welcome frame

# Tkinter setup
root = tk.Tk()
root.title("Gesture Recognition")
root.geometry("1024x768")  # Enlarged Tkinter window size

# Style configurations
bg_color = "#282c34"  # Dark background color
btn_color = "#61afef"  # Button color
btn_hover_color = "#98c379"  # Button hover color
text_color = "#abb2bf"  # Text color

# Welcome Page
welcome_frame = tk.Frame(root, bg=bg_color)
welcome_frame.pack(fill=tk.BOTH, expand=True)

app_name_label = tk.Label(welcome_frame, text="Gesture Recognition", font=("Helvetica", 24), bg=bg_color, fg=text_color)
app_name_label.pack(pady=20)

student_name_label = tk.Label(welcome_frame, text="Team Members: Arshad, Ayaan", font=("Helvetica", 16), bg=bg_color, fg=text_color)
student_name_label.pack(pady=10)

usn_label = tk.Label(welcome_frame, text="USN: 1CR21CS039, 1CR21CS44", font=("Helvetica", 16), bg=bg_color, fg=text_color)
usn_label.pack(pady=10)

start_button = tk.Button(welcome_frame, text="Enter Application", command=show_main_app, font=("Helvetica", 16), bg=btn_color, fg=bg_color, activebackground=btn_hover_color, relief=tk.RAISED)
start_button.pack(pady=20)

# Main Application Page
main_frame = tk.Frame(root, bg=bg_color)

# Back to Welcome Screen Button
back_button = tk.Button(main_frame, text="Back to Welcome Screen", command=show_welcome_screen, font=("Helvetica", 16), bg=btn_color, fg=bg_color, activebackground=btn_hover_color, relief=tk.RAISED)
back_button.grid(row=0, column=0, padx=10, pady=10)

# Start Recognition Button
start_button = tk.Button(main_frame, text="Start Recognition", command=start_recognition, width=20, height=2, font=("Helvetica", 16), bg=btn_color, fg=bg_color, activebackground=btn_hover_color, relief=tk.RAISED)
start_button.grid(row=0, column=1, padx=10, pady=10)

# Stop Recognition Button
stop_button = tk.Button(main_frame, text="Stop Recognition", command=stop_recognition, width=20, height=2, font=("Helvetica", 16), bg=btn_color, fg=bg_color, activebackground=btn_hover_color, relief=tk.RAISED)
stop_button.grid(row=0, column=2, padx=10, pady=10)

# Video Feed Label
video_label = tk.Label(main_frame)
video_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

# Status Label
status_label = tk.Label(main_frame, text="Status: Ready", font=("Helvetica", 12), bg=bg_color, fg=text_color)
status_label.grid(row=2, column=0, columnspan=3, pady=20)

# Configure grid row and column weights
main_frame.grid_rowconfigure(1, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_columnconfigure(2, weight=1)

# Load model weights
_load_weights()

# Start the Tkinter main loop
root.mainloop()
