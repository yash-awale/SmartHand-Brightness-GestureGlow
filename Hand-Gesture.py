import cv2
import mediapipe as mp
import screen_brightness_control as sbc

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    # Flip and convert color
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb tip: 4, Index tip: 8
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Calculate distance between thumb and index tip
            height, width, _ = image.shape
            x1, y1 = int(thumb_tip.x * width), int(thumb_tip.y * height)
            x2, y2 = int(index_tip.x * width), int(index_tip.y * height)

            distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

            # Map the distance to brightness level (distance range approx: 20 to 200)
            brightness = int(max(0, min(100, (distance - 20) * 1)))  # Adjust scale if needed

            # Set brightness
            try:
                sbc.set_brightness(brightness)
            except Exception as e:
                print("Brightness Error:", e)

            # Show brightness value on screen
            cv2.putText(image, f'Brightness: {brightness}%', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show output
    cv2.imshow("Hand Gesture Brightness Control", image)

    # 🔑 Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
