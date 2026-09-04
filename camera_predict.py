import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# ======================
# 🔌 Arduino Setup
# ======================
# Replace 'COM3' with your Arduino port (e.g., 'COM4' or '/dev/ttyUSB0' on Linux)
# arduino = serial.Serial('COM4', 9600)
# time.sleep(2)  # Wait for Arduino to initialize

# ======================
# 🧠 Load Model
# ======================
model = tf.keras.models.load_model("trash_classifier.h5")

# Class labels (same order as training)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ======================
# 📷 Camera Setup
# ======================
cap = cv2.VideoCapture(0)  # 0 = default camera
img_height, img_width = 150, 150

last_label = None  # To avoid sending the same label repeatedly

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Preprocess frame for prediction
    img = cv2.resize(frame, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 150, 150, 3)

    # Predict
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    label = class_labels[class_index]
    confidence = predictions[0][class_index]

    # Display on video
    cv2.putText(frame, f"{label} ({confidence*100:.2f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Trash Classifier", frame)

    # ======================
    # 📤 Send label to Arduino
    # ======================
    # Only send if label changes or confidence > 80%
    if label != last_label and confidence > 0.8:
        try:
            arduino.write(label.encode())
            arduino.write(b'\n')  # newline marks end of message
            print(f"➡ Sent to Arduino: {label}")
            last_label = label
        except Exception as e:
            print(f"⚠️ Error sending to Arduino: {e}")

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# 🔚 Cleanup
# ======================
cap.release()
arduino.close()
cv2.destroyAllWindows()
