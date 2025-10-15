
# 🎗️ Breast Cancer Classification App

<div align="center">
  <img src="https://placehold.co/800x200/F0F2F6/333333?text=Breast+Cancer+Classifier&font=inter" alt="Breast Cancer Classifier Banner">
</div>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python">
  <img alt="Framework" src="https://img.shields.io/badge/Streamlit-1.30.0-ff4b4b.svg?style=for-the-badge&logo=streamlit">
  <img alt="ML Library" src="https://img.shields.io/badge/scikit--learn-1.4.2-f7931e.svg?style=for-the-badge&logo=scikit-learn">
</p>

---

## 🧠 Overview
The **Breast Cancer Classification App** is an interactive machine learning web application built using **Streamlit**.  
It leverages a **Logistic Regression** model trained on real-world breast cancer data to predict whether a tumor is **benign** or **malignant** based on 30 key input features.

This project demonstrates a practical application of **data science, machine learning, and web deployment** — all within a user-friendly interface.

---

## ✨ Features

- **🕒 Real-time Predictions:** Get instant classification results as you adjust feature sliders.  
- **🎚️ Interactive UI:** Clean, responsive interface using Streamlit’s intuitive components.  
- **📊 Confidence Score:** Displays model confidence in predictions.  
- **🧩 Self-contained:** Lightweight app — easy to deploy anywhere.  
- **🎨 Custom Styling:** Polished interface using a separate CSS file for visual appeal.

---

## 📸 Application Preview

Below is a glimpse of the application layout.  
The sidebar contains sliders for all 30 tumor features, and the main section displays the classification result and confidence score in real time.

> 🖼️ *Replace this placeholder with a screenshot of your running app.*

<div align="center">
  <img src="app_screenshot.png" alt="Breast Cancer App Screenshot" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
</div>
<img width="1920" height="912" alt="{1BA124DB-4BAA-4771-82AD-1B544C65E29C}" src="https://github.com/user-attachments/assets/9ec4c1ce-e171-4641-8cbc-1139887df46d" />

---

## 📂 Project Structure

```bash
.
├── 📄 app.py              # Main Streamlit application script
├── 🎨 style.css           # Custom styling for the UI
├── 🧠 best_LR.pkl         # Pre-trained Logistic Regression model
├── 📋 requirements.txt    # Project dependencies
└── 📖 README.md           # Project documentation
