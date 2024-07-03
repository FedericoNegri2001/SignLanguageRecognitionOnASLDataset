import os
import pickle
import mediapipe as mp
import cv2

# Mediapipe objects for hand landmark detection and drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands with static image mode and a minimum detection confidence of 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = 'path of dataset'

data = []
labels = []

# Loop through each directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the current directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        # Read and convert the image to RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Loop through each detected hand's landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Store x and y coordinates separately
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Normalize the coordinates
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        # Normalize the landmark coordinates to a fixed size
        if len(data_aux) < 84:
            data_aux += [0.0] * (84 - len(data_aux))  # Pad with zeros if less than 84
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]  # Truncate to 84 if more than 84

        # Append to data and labels lists if the data_aux length is 84
        if len(data_aux) == 84:
            data.append(data_aux)
            labels.append(dir_)

# Save data and labels to a pickle file
with open('data_with_landmarks.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
