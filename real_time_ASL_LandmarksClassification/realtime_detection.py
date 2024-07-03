import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model_rf_90.p', 'rb'))
model = model_dict['model']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame from the webcam
    ret, frame = cap.read()

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extract landmark coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Normalize the landmark coordinates to a fixed size
        if len(data_aux) < 84:
            data_aux += [0.0] * (84 - len(data_aux))  # Pad with zeros if less than 84
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]  # Truncate to 84 if more than 84

        # Make a prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = prediction[0]  # Get the predicted character

        # Draw bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
