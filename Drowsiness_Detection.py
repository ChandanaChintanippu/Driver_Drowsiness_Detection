from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import pyttsx3

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Function to speak a message
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Initialize the music mixer for alert sound
mixer.init()
mixer.music.load(r"C:\Drivers_Drowsiness_Detection\music.wav")

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Drowsiness detection thresholds
thresh = 0.25
frame_check = 20
flag = 0

# Load dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Drivers_Drowsiness_Detection\models-20240910T173130Z-001\models\shape_predictor_68_face_landmarks.dat")

# Define facial landmarks for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Average EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw contours around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Drowsiness detection logic
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "ALERT! Drowsiness Detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                speak("Wake up!")  # Voice alert
                print("Drowsiness detected!")
                if not mixer.music.get_busy():  # Check if sound is not already playing
                    mixer.music.play()
        else:
            flag = 0

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    
    # Exit condition for the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
