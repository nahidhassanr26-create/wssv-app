 import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import datetime
from google import genai  # নতুন ফ্রি এআই লাইব্রেরি

# 🔒 সরাসরি টোকেন না লিখে স্ট্রিমলিটের সিক্রেট ম্যানেজার ব্যবহার করা হলো
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# পেজ সেটিংস
st.set_page_config(page_title="WSSV ডিটেকশন ও পরামর্শ কেন্দ্র", page_icon="🦐", layout="centered")

@st.cache_resource
def load_model():
    return YOLO('best.tflite')

try:
    model = load_model()
except Exception as e:
    st.error("মডেল লোড হতে সমস্যা হচ্ছে। 'best.tflite' ফাইলটি সঠিক জায়গায় আছে কিনা চেক করুন।")

st.title("🦐 WSSV (হোয়াইট স্পট) ডিটেকশন ও পরামর্শ কেন্দ্র")
st.write("আপনার চিংড়ির ছবি আপলোড করে বা সরাসরি摄像头 দিয়ে তুলে ২৪/৭ পরীক্ষা করুন।")

uploaded_file = st.file_uploader("চিংড়ির ছবি নির্বাচন করুন...", type=["jpg", "jpeg", "png"])
cam_file = st.camera_input("অথবা সরাসরি ছবি তুলুন")

image = None
if uploaded_file is not None: 
    image = Image.open(uploaded_file)
elif cam_file is not None: 
    image = Image.open(cam_file)

# পিডিএফ জেনারেটর ফাংশন
def generate_pdf(status, advice, spot_cnt, severity, conf_pct, img_array):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#1E3A8A'), spaceAfter=15)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=12, leading=16, spaceAfter=10)
    alert_style = ParagraphStyle('Alert', parent=styles['Normal'], fontSize=12, leading=16, textColor=colors.HexColor('#B91C1C'), spaceAfter=10)
    
    story.append(Paragraph("🦐 WSSV Detection Report", title_style))
    story.append(Paragraph(f"Date: {datetime.date.today().strftime('%B %d, %Y')}", body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph(f"<b>Status:</b> {status}", alert_style if spot_cnt > 0 else body_style))
    story.append(Paragraph(f"<b>Detected Spots:</b> {spot_cnt} items", body_style))
    story.append(Paragraph(f"<b>Risk Level:</b> {severity}", body_style))
    story.append(Paragraph(f"<b>Confidence:</b> {conf_pct}%", body_style))
    story.append(Spacer(1, 15))
    
    rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='JPEG', quality=80)
    img_buffer.seek(0)
    
    rl_img = RLImage(img_buffer, width=300, height=300)
    story.append(rl_img)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Actionable Advice / Recommendations:</b>", ParagraphStyle('Sub', parent=styles['Heading2'], fontSize=14, spaceAfter=8)))
    
    if spot_cnt > 0:
        adv_text = "1. Stop water exchange immediately to prevent spreading.<br/>2. Conduct emergency harvest if shrimps are near marketable size.<br/>3. Disinfect nets and tools with chlorine water before using in other ponds.<br/>4. Immediately contact your local sub-district fisheries officer."
    else:
        adv_text = "1. Maintain strict farm biosecurity.<br/>2. Monitor water parameters (pH, salinity, ammonia) regularly.<br/>3. Test post-larvae (PL) before stocking new batches."
    
    story.append(Paragraph(adv_text, body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

if image is not None:
    st.image(image, caption='আপনার দেওয়া ছবি', use_container_width=True)
    
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    if st.button("পরীক্ষা করুন (Analyze)", type="primary") or st.session_state.processed:
        st.session_state.processed = True
        
        results = model(image, conf=0.05)
        spot_count = 0
        max_conf = 0.0
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for r in results:
            spot_count = len(r.boxes)
            if spot_count > 0:
                max_conf = np.max(r.boxes.conf.cpu().numpy())
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    if score >= 0.05:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"WSSV: {score:.2f}"
                        cv2.putText(img_cv, label, (x1, max(y1 - 10, 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        output_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.subheader("📊 পরীক্ষার ফলাফল:")
        st.image(output_image, use_container_width=True)
        
        conf_percentage = int(max_conf * 100)
        
        if spot_count > 0 and max_conf >= 0.05:
            severity = "কম ঝুঁকি (Low)" if spot_count <= 2 else "মাঝারি ঝুঁকি (Medium)" if spot_count <= 5 else "উচ্চ ঝুঁকি (High Risk)"
            status_title = "WSSV Risk Detected"
            st.error(f"🔴 WSSV (হোয়াইট স্পট) আক্রান্ত হওয়ার ঝুঁকি পাওয়া গেছে!")
            st.write(f"* **চিহ্নিত লক্ষণের সংখ্যা:** {spot_count} টি")
            st.warning(f"🚨 রিস্ক লেভেল: {severity}")
        else:
            severity = "No Risk"
            status_title = "Healthy Shrimp"
            st.success("🟢 কোনো উল্লেখযোগ্য WSSV ঝুঁকি পাওয়া যায়নি (সুস্থ চিংড়ি)")
            
        pdf_data = generate_pdf(status_title, "", spot_count, severity, conf_percentage, img_cv)
        st.download_button(
            label="📥 পরীক্ষার পিডিএফ রিপোর্ট ডাউনলোড করুন",
            data=pdf_data,
            file_name=f"WSSV_Report.pdf",
            mime="application/pdf"
        )

# 🤖 ২. ফ্রি এআই খামার উপদেষ্টা চ্যাটবট সেকশন
st.write("---")
st.header("🤖 🦐 এআই চিংড়ি খামার উপদেষ্টা (AI Advisor)")
st.write("চিংড়ি চাষ, পুকুরের পানি ব্যবস্থাপনা বা যেকোনো রোগ নিয়ে বাংলায় প্রশ্ন করুন।")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, text in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

user_query = st.chat_input("আপনার প্রশ্নটি এখানে লিখুন (যেমন: চিংড়ির ঘেরে অ্যামোনিয়া কমাবো কেমনে?)...")

if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append(("user", user_query))
    
    with st.spinner("এআই উত্তর তৈরি করছে..."):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            system_prompt = "You are an expert aquaculture scientist and shrimp farming advisor in Bangladesh. Answer the user's questions accurately in Bengali language."
            full_prompt = f"{system_prompt}\nUser Question: {user_query}"
            
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=full_prompt,
            )
            
            ai_response = response.text
            st.chat_message("assistant").write(ai_response)
            st.session_state.chat_history.append(("assistant", ai_response))
        except Exception as e:
            st.error("দুঃখিত, চ্যাটবট সচল করতে সমস্যা হচ্ছে। অনুগ্রহ করে Streamlit Settings এ গিয়ে Secrets ইনপুট দিন।")
