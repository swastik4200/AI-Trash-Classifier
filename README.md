# AI-Trash-Classifier
# ♻️ AI Waste Classification System

An AI-powered real-time waste classification system using **TensorFlow/Keras, OpenCV, and CNN** to classify waste into six categories. The system uses a webcam for real-time image classification and can be integrated with **Arduino** for automated waste segregation.

## 📌 Overview

This project uses a Convolutional Neural Network (CNN) trained on a waste image dataset to classify different types of waste.

The system currently supports six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The webcam captures live video, extracts a predefined **Region of Interest (ROI)**, preprocesses the image, and sends it to the trained CNN model for classification.

The predicted waste category can also be sent to an **Arduino through serial communication**, allowing the system to be extended into an automated physical waste-segregation system.

---

## ✨ Features

- 🎥 Real-time waste classification using a webcam
- 🤖 CNN-based image classification using TensorFlow/Keras
- ♻️ Classification into 6 waste categories
- 🎯 Region of Interest (ROI) based prediction
- 📊 Real-time prediction confidence display
- 🔌 Arduino serial communication support
- 🖼️ Image preprocessing using OpenCV and NumPy
- 💾 Trained model saved in `.h5` format

---

## 🧠 Waste Categories

| Category | Description |
|---|---|
| Cardboard | Cardboard and paperboard materials |
| Glass | Glass bottles and containers |
| Metal | Metal cans and objects |
| Paper | Paper-based waste |
| Plastic | Plastic bottles and containers |
| Trash | General/other waste |

---

## 🏗️ System Architecture

```text
                ┌─────────────────┐
                │  Laptop Webcam  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  ROI Extraction │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Image Preprocess│
                │ Resize + Scale  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    CNN Model    │
                │   TensorFlow    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Waste Prediction│
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
       ┌────────────────┐  ┌────────────────┐
       │ Display Result │  │    Arduino     │
       │ + Confidence   │  │ Serial Output  │
       └────────────────┘  └───────┬────────┘
                                   │
                                   ▼
                         Automated Segregation

🛠️ Technologies Used
Python
TensorFlow
Keras
OpenCV
NumPy
PySerial
Arduino
Convolutional Neural Network (CNN)
