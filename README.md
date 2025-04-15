# 🧠 Resume Job Role Classifier

An automated system that classifies resumes into specific job roles (like HR, Software Developer, Accountant, etc.) using traditional ML models and BERT, with a Streamlit-powered frontend that accepts resumes in PDF format.

---

## 🚀 Project Overview

This project aims to streamline resume processing by automatically classifying resumes into relevant job roles. The workflow combines data preprocessing, multiple machine learning techniques, and a user-friendly interface.

---

## 🛠️ Project Workflow

1. **Data Loading**
   - Loaded resume dataset from Kaggle.
  
2. **Data Cleaning**
   - Removed noise, HTML tags, special characters.
   - Normalized text (lowercasing, stopword removal, etc.).

3. **Data Augmentation**
   - Enhanced dataset diversity using techniques like synonym replacement, etc.

4. **Vectorization**
   - Converted textual data into numerical form using:
     - Word2Vec
     - TF-IDF

5. **Model Pre-Testing**
   - **K-Nearest Neighbors (KNN)**
     - Evaluated performance on preprocessed vectors.
   - **Traditional ML Models**
     - Logistic Regression
     - Decision Tree
     - Random Forest

6. **Final Classification**
   - Used **BERT** for contextual understanding and accurate classification of job roles.

7. **Frontend with Streamlit**
   - Simple, interactive interface.
   - Users upload resume (PDF format).
   - Resume is parsed and classified into a job category.

---

## 💻 Tech Stack

- **Languages:** Python
- **ML/NLP:** scikit-learn,Word2Vec, Transformers (Hugging Face BERT)
- **Frontend:** Streamlit
- **PDF Parsing:** PyMuPDF 


---

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/resume-role-classifier.git
   cd resume-role-classifier
