import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("resume_model")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

def classify_resume(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

st.title("Resume Category Classifier")

uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()

    if resume_text:
        st.subheader("Extracted Resume Text")
        st.write(resume_text[:1000] + "...")  

        if st.button("Predict Category"):
            prediction = classify_resume(resume_text)
            
            label_map = {
                0: "HR",
                1: "DESIGNER",
                2: "INFORMATION-TECHNOLOGY",
                3: "TEACHER",
                4: "ADVOCATE",
                5: "BUSINESS-DEVELOPMENT",
                6: "HEALTHCARE",
                7: "FITNESS",
                8: "AGRICULTURE",
                9: "BPO",
                10: "SALES",
                11: "CONSULTANT",
                12: "DIGITAL-MEDIA",
                13: "AUTOMOBILE",
                14: "CHEF",
                15: "FINANCE",
                16: "APPAREL",
                17: "ENGINEERING",
                18: "ACCOUNTANT",
                19: "CONSTRUCTION",
                20: "SOFTWARE",
                21: "BANKING",
                22: "ARTS",
                23: "AVIATION"
            }

            st.success(f"Predicted Category: {label_map.get(prediction, 'ENGINEERING')}")
    else:
        st.error("Could not extract text from the PDF.")
