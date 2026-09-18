# 🤖 AI Career Placement Predictor

An AI/ML-based web application that predicts a student's placement probability based on academic performance, technical skills, aptitude, internships, certifications, projects, and other relevant factors.

## 👨‍💻 Developer

**Gaurav Jain**  
B.Tech CSE (AIML)  
Poornima University

---

## 📌 Project Overview

The **AI Career Placement Predictor** is designed to help students understand their placement prospects based on their current academic and skill profile.

The project uses a **Machine Learning classification model** to predict whether a student is likely to be placed or not placed.

The application also provides:

- Placement probability
- Important factors affecting the prediction
- Personalized improvement suggestions
- 30-day preparation plan
- Dataset analytics and visualizations
- Student-level data exploration

---

## 🎯 Objectives

- Predict student placement outcomes using Machine Learning.
- Analyze the relationship between skills and placement.
- Help students identify areas for improvement.
- Provide an interactive and easy-to-use prediction interface.
- Demonstrate an end-to-end AI/ML project workflow.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib
- Git & GitHub

---

## 📊 Dataset

The project uses an Indian student placement dataset containing **12,000 student records** and **16 columns**.

### Important Features

- Gender
- Age
- Degree
- Branch
- CGPA
- Backlogs
- Internships
- Certifications
- Coding Skills
- Communication Skills
- Aptitude Score
- Projects

### Target

`placed`

The model predicts whether the student is:

- **Placed**
- **Not Placed**

Post-placement fields such as company type and package information are not used as prediction inputs.

---

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier**.

### Model Pipeline

1. Load dataset
2. Select relevant features
3. Preprocess categorical features using One-Hot Encoding
4. Keep numerical features as numerical values
5. Split data into training and testing sets
6. Train Random Forest Classifier
7. Evaluate model
8. Save trained model as `placement_model.pkl`
9. Use the saved model in the Streamlit application

### Model Configuration

```text
Algorithm: Random Forest Classifier
Number of Estimators: 200
Random State: 42
Test Size: 20%