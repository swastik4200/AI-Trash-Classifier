import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# ======================
# 🔌 Arduino Setup
# ======================
# ⚠️ Change COM4 to your Arduino's actual port (e.g., COM4 or /dev/ttyUSB0)
# arduino = serial.Serial('COM4', 9600)
# time.sleep(2)  # Wait for Arduino to initialize

# ======================
# 🧠 Load Trained Model
# ======================
model = tf.keras.models.load_model("trash_classifier.h5")
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ======================
# 📷 Camera Setup
# ======================
cap = cv2.VideoCapture(0)
img_height, img_width = 150, 150

# ROI coordinates (adjust as needed)
# These define the rectangle area where trash should be placed
x_start, y_start, x_end, y_end = 200, 100, 440, 340

last_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Draw ROI rectangle
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)
    cv2.putText(frame, "Place trash here", (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Extract ROI only
    roi = frame[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        continue

    # Preprocess ROI for prediction
    img = cv2.resize(roi, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict only inside ROI
    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions[0])
    label = class_labels[class_index]
    confidence = predictions[0][class_index]

    # Display prediction
    cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ======================
    # 📤 Send to Arduino (only if confident)
    # ======================
    if label != last_label and confidence > 0.8:
        try:
            arduino.write(label.encode())
            arduino.write(b'\n')
            print(f"➡ Sent to Arduino: {label}")
            last_label = label
        except Exception as e:
            print(f"⚠️ Serial error: {e}")

    # Show camera feed
    cv2.imshow("Trash Classifier (ROI Mode)", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# 🔚 Cleanup
# ======================
cap.release()
arduino.close()
cv2.destroyAllWindows()
