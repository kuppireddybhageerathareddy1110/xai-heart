# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
import uuid
import os
import time
from lime.lime_tabular import LimeTabularExplainer

# OCR
import pytesseract
from PIL import Image
import cv2
from pypdf import PdfReader
import io
from pdf2image import convert_from_bytes

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(layout="wide")
st.title("🫀 Multimodal Cardiac Clinical Decision Support AI")

# =========================================================
# IMAGE SAVE HELPER (CRITICAL)
# =========================================================
def save_fig(fig,name):
    path=os.path.abspath(name)
    fig.savefig(path,dpi=200,bbox_inches="tight")
    plt.close(fig)
    time.sleep(0.2)
    return path

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    mit=tf.keras.models.load_model("models/ecg_cnn.h5",compile=False)
    ptb=tf.keras.models.load_model("models/ptbxl_encoder.keras",compile=False)

    clin=xgb.XGBClassifier()
    clin.load_model("models/clinical_xgb.json")

    fusion=joblib.load("models/multimodal_heart_ai.pkl")
    if not isinstance(fusion,list):
        fusion=[fusion]

    cols=joblib.load("models/clinical_features.pkl")
    return mit,ptb,clin,fusion,cols

mit_model,ptb_encoder,clin_model,fusion_models,feature_cols=load_models()


# =========================================================
# BACKGROUND POPULATION (MUST BE HERE)
# =========================================================
@st.cache_data
def background_population():

    n = 1200

    age = np.random.normal(55, 12, n).clip(25,85)
    chol = np.random.normal(240, 45, n).clip(120,380)
    oldpeak = np.abs(np.random.normal(1.2, 1.1, n)).clip(0,6)
    exang = np.random.binomial(1,0.35,n)
    male = np.random.binomial(1,0.68,n)

    df = pd.DataFrame({
        "age":age,
        "chol":chol,
        "oldpeak":oldpeak,
        "exang_True":exang,
        "sex_Male":male
    })

    for c in feature_cols:
        if c not in df.columns:
            df[c]=0

    return df[feature_cols]


# IMPORTANT — NOW SAFE
bg_data = background_population()


# =========================================================
# UTILITIES
# =========================================================
def safe_prob(p): return float(np.clip(p,0,1))
def normalize(s): return (s-np.mean(s))/(np.std(s)+1e-6)

def risk_category(p):
    if p<0.25:return "LOW RISK"
    if p<0.60:return "MODERATE RISK"
    return "HIGH RISK"

# =========================================================
# GRADCAM
# =========================================================
def gradcam(signal):
    x=tf.convert_to_tensor(signal[np.newaxis,...,np.newaxis],dtype=tf.float32)
    last=None
    with tf.GradientTape() as tape:
        tape.watch(x)
        out=x
        for layer in mit_model.layers:
            out=layer(out)
            if isinstance(layer,tf.keras.layers.Conv1D):
                last=out
        loss=out[:,tf.argmax(out[0])]
    grads=tape.gradient(loss,last)
    w=tf.reduce_mean(grads,axis=1)
    cam=tf.reduce_sum(w*last,axis=-1)[0]
    cam=tf.maximum(cam,0);cam/=tf.reduce_max(cam)+1e-8
    return np.interp(np.linspace(0,len(cam)-1,len(signal)),np.arange(len(cam)),cam.numpy())


# =========================================================
# XAI PLOTS
# =========================================================
def shap_plot(df):
    explainer=shap.TreeExplainer(clin_model)
    sv=explainer.shap_values(df)
    fig=plt.figure(figsize=(7,4))
    shap.plots.waterfall(shap.Explanation(values=sv[0],base_values=explainer.expected_value,data=df.iloc[0]),show=False)
    return save_fig(fig,"shap.png")

def lime_plot(df):

    explainer = LimeTabularExplainer(
        training_data = bg_data.values,
        feature_names = list(bg_data.columns),
        class_names = ["Healthy","Heart Disease"],
        mode = "classification",
        discretize_continuous = True,
        sample_around_instance = True
    )

    exp = explainer.explain_instance(
        df.values[0],
        clin_model.predict_proba,
        num_features = min(8,len(feature_cols)),
        num_samples = 1500
    )

    fig = exp.as_pyplot_figure()
    return save_fig(fig,"lime.png")

def pdp_plot(df,feature):
    values=np.linspace(0,400,30)
    preds=[]
    for v in values:
        temp=df.copy()
        temp[feature]=v
        preds.append(clin_model.predict_proba(temp)[0,1])
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(values,preds)
    ax.set_title("Population Risk Response")
    return save_fig(fig,"pdp.png")

# =========================================================
# OCR
# =========================================================
def preprocess(img):
    img=np.array(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(gray)
    th=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,5)
    return cv2.medianBlur(th,3)

def extract_text(file):
    text=""
    if file.type=="application/pdf":
        pdf=PdfReader(io.BytesIO(file.read()))
        for page in pdf.pages:
            t=page.extract_text()
            if t:text+=t
        if len(text)>50:return text.lower()
        file.seek(0)
        pages=convert_from_bytes(file.read(),dpi=300)
        for p in pages:
            text+=pytesseract.image_to_string(preprocess(p))
    else:
        img=Image.open(file).convert("RGB")
        text=pytesseract.image_to_string(preprocess(img))
    return text.lower()
# =========================================================
# OCR INTERPRETATION (IMPORTANT)
# =========================================================
def detect_conditions(text):

    text = text.lower()
    findings = set()

    # -------------------------
    # RISK INTERPRETATION
    # -------------------------
    if "low risk" in text or "no clinically significant abnormality" in text:
        findings.add("normal")

    if "moderate risk" in text:
        findings.add("borderline")

    if "high risk" in text:
        findings.add("abnormal")

    # -------------------------
    # ECG DIAGNOSIS TERMS
    # -------------------------
    cardiac_terms = {
        "ischemia":[
            "st elevation","st depression","t wave inversion","infarct","ischemia"
        ],
        "arrhythmia":[
            "atrial fibrillation","irregular rhythm","afib","pvc","pac"
        ],
        "tachycardia":[
            "tachycardia","sinus tachycardia"
        ],
        "bradycardia":[
            "bradycardia","sinus bradycardia"
        ]
    }

    for label,words in cardiac_terms.items():
        for w in words:
            if w in text:
                findings.add(label)

    # -------------------------
    # AI GENERATED REPORT DETECTION
    # -------------------------
    if "gradcam" in text or "shap" in text or "lime" in text or "pdp" in text:
        findings.add("ai_analysis_present")

    if len(findings) == 0:
        return ["uncertain"]

    return list(findings)

# =========================================================
# ECG INPUT
# =========================================================
st.header("1️⃣ ECG Analysis")
file=st.file_uploader("Upload ECG (.npy)",type=["npy"])
ecg_risk=None;ptb_features=None

if file:
    signal=normalize(np.load(file))
    ecg_risk=safe_prob(1-mit_model.predict(signal[np.newaxis,...,np.newaxis])[0,0])

    cam=gradcam(signal)
    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot(signal);ax.plot(cam*signal.max(),color='red')
    gradcam_img=save_fig(fig,"gradcam.png")
    st.image(gradcam_img)

    sig=np.interp(np.linspace(0,len(signal)-1,1000),np.arange(len(signal)),signal)
    ptb_features=ptb_encoder.predict(np.tile(sig.reshape(1000,1),(1,12))[np.newaxis,...])[0]

# =========================================================
# CLINICAL INPUT
# =========================================================
st.header("2️⃣ Clinical Factors")
age=st.slider("Age",20,90,50)
chol=st.slider("Cholesterol",100,400,200)
oldpeak=st.slider("ST depression",0.0,6.0,1.0)
exang=st.selectbox("Exercise Angina",[0,1])
male=st.selectbox("Male",[0,1])

df=pd.DataFrame(columns=feature_cols);df.loc[0]=0
for c,v in {"age":age,"chol":chol,"oldpeak":oldpeak,"exang_True":exang,"sex_Male":male}.items():
    if c in df.columns:df.loc[0,c]=v

clin_prob=safe_prob(clin_model.predict_proba(df)[0,1])
st.write("Clinical risk:",round(clin_prob,3))

shap_img=shap_plot(df); st.image(shap_img)
lime_img=lime_plot(df); st.image(lime_img)
pdp_img=pdp_plot(df,"chol"); st.image(pdp_img)

# =========================================================
# FINAL DECISION + REPORT
# =========================================================
st.header("3️⃣ Final Decision")

if ecg_risk is not None:
    fusion_vec=np.concatenate([[clin_prob],[ecg_risk],ptb_features]).reshape(1,-1)
    final=safe_prob(np.mean([m.predict_proba(fusion_vec)[0,1] for m in fusion_models]))

    st.subheader(risk_category(final))
    st.write("Risk score:",round(final,3))

    def generate_pdf():
        name=f"cardiac_report_{uuid.uuid4().hex}.pdf"
        doc=SimpleDocTemplate(name)
        styles=getSampleStyleSheet()

        story=[
            Paragraph("AI Cardiac Clinical Decision Report",styles['Title']),
            Spacer(1,20),
            Paragraph(f"Risk Score: {final:.3f}",styles['Normal']),
            Paragraph(f"Category: {risk_category(final)}",styles['Normal']),
            Spacer(1,20)
        ]

        for title,path in {
            "ECG GradCAM":gradcam_img,
            "SHAP":shap_img,
            "LIME":lime_img,
            "PDP":pdp_img
        }.items():
            story.append(Paragraph(title,styles['Heading2']))
            story.append(Spacer(1,10))
            story.append(PDFImage(path,width=6*inch,height=3*inch))
            story.append(Spacer(1,20))

        doc.build(story)
        return name

    if st.button("📄 Generate Clinical Report"):
        pdf_file=generate_pdf()
        with open(pdf_file,"rb") as f:
            st.download_button("⬇ Download Report",f,"AI_Cardiac_Report.pdf")
# =========================================================
# OCR SECOND OPINION
# =========================================================
st.header("4️⃣ Upload Hospital Report")
report = st.file_uploader("Upload report", type=["png","jpg","jpeg","pdf"])

if report and ecg_risk is not None:

    # ---------- Extract ----------
    txt = extract_text(report)

    st.subheader("Extracted Text")
    st.text(txt[:1500])

    # ---------- Detect ----------
    doc = detect_conditions(txt)
    st.subheader("Detected diagnosis")
    st.write(doc)

    # ---------- Compare with AI ----------
    st.subheader("AI Second Opinion")

    if "normal" in doc and final < 0.25:
        st.success("AI agrees with report — ECG appears normal")

    elif "normal" in doc and final >= 0.25:
        st.warning("⚠ AI suspects hidden abnormality")

    elif "abnormal" in doc and final < 0.25:
        st.warning("⚠ AI believes ECG safer than reported")

    elif "ai_analysis_present" in doc:
        st.info("Uploaded document appears AI-generated, not physician interpreted")

    else:
        st.info("No clear clinical conclusion detected")
