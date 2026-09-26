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

# ১. পেজ সেটিংস
st.set_page_config(page_title="WSSV ডিটেকশন ও পরামর্শ কেন্দ্র", page_icon="🦐", layout="centered")

@st.cache_resource
def load_model():
    return YOLO('best.tflite')

try:
    model = load_model()
except Exception as e:
    st.error("মডেল লোড হতে সমস্যা হচ্ছে। 'best.tflite' ফাইলটি সঠিক জায়গায় আছে কিনা চেক করুন।")

st.title("🦐 WSSV (হোয়াইট স্পট) ডিটেকশন ও পরামর্শ কেন্দ্র")
st.write("আপনার চিংড়ির ছবি আপলোড করে বা সরাসরি ক্যামেরা দিয়ে তুলে ২৪/৭ পরীক্ষা করুন।")

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
            st.markdown("""
            **🚨 তাত্ক্ষণিক জরুরি পরামর্শ (Actionable Advice):**
            1. **পানি বিনিময় বন্ধ করুন:** এই পুকুর বা ঘেরের পানি অন্য কোথাও ছড়াতে দেবেন না।
            2. **Emergency Harvest:** চিংড়ি বিক্রির উপযোগী সাইজের কাছাকাছি হলে দ্রুত ধরে ফেলুন, নয়তো ব্যাপক মড়ক হতে পারে।
            3. **বিশেষজ্ঞের সহায়তা:** অবিলম্বে আপনার নিকটস্থ উপজেলা মৎস্য কর্মকর্তা বা অ্যাকুয়াকালচার ল্যাবের সাথে যোগাযোগ করুন।
            """)
        else:
            severity = "No Risk"
            status_title = "Healthy Shrimp"
            st.success("🟢 কোনো উল্লেখযোগ্য WSSV ঝুঁকি পাওয়া যায়নি (সুস্থ চিংড়ি)")
            st.markdown("""
            **📋 সাধারণ পরামর্শ:**
            * খামারের সাধারণ বায়োসিকিউরিটি বজায় রাখুন।
            * নিয়মিত পানির গুণাগুণ পরীক্ষা করুন।
            """)
            
        pdf_data = generate_pdf(status_title, "", spot_count, severity, conf_percentage, img_cv)
        st.write("---")
        st.download_button(
            label="📥 পরীক্ষার পিডিএফ রিপোর্ট ডাউনলোড করুন",
            data=pdf_data,
            file_name=f"WSSV_Report.pdf",
            mime="application/pdf"
        )

# 📞 ৩. নতুন উপজেলা ভিত্তিক মৎস্য কর্মকর্তা ডিরেক্টরি সেকশন
st.write("---")
st.header("📞 উপজেলা মৎস্য কর্মকর্তা ডিরেক্টরি (Helpline)")
st.write("আক্রান্ত ফলাফল পাওয়া গেলে দ্রুত আপনার এলাকার কর্মকর্তার সাথে যোগাযোগ করুন:")

# মৎস্য কর্মকর্তাদের ডিরেক্টরি ডেটাবেস
officer_data = {
    "বাগেরহাট (Bagerhat)": {
        "বাগেরহাট সদর": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৬৮\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, বাগেরহাট সদর।",
        "মোংলা": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৭০\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, মোংলা, বাগেরহাট।",
        "শরণখোলা": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৭৫\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, শরণখোলা, বাগেরহাট।",
        "রামপাল": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯१৭১\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, রামপাল, বাগেরহাট।"
    },
    "সাতক্ষীরা (Satkhira)": {
        "সাতক্ষীরা সদর": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৫৮\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, সাতক্ষীরা সদর।",
        "শ্যামনগর": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৬৪\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, শ্যামনগর, সাতক্ষীরা।",
        "কালিগঞ্জ": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৬৩\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, কালিগঞ্জ, সাতক্ষীরা।",
        "আশাশুনি": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৬১\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, আশাশুনি, সাতক্ষীরা।"
    },
    "খুলনা (Khulna)": {
        "খুলনা সদর/ডুমুরিয়া": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৫০\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, ডুমুরিয়া, খুলনা।",
        "কয়রা": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৫৪\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, কয়রা, খুলনা।",
        "পাইকগাছা": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৫৩\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, পাইকগাছা, খুলনা।",
        "দাকোপ": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৫২\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, দাকোপ, খুলনা।"
    },
    "কক্সবাজার (Cox's Bazar)": {
        "কক্সবাজার সদর": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৪০\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, কক্সবাজার সদর।",
        "টেকনাফ": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৪৭\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, টেকনাফ, কক্সবাজার।",
        "উখিয়া": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৪৬\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, উখিয়া, কক্সবাজার।",
        "চকুড়িয়া": "নাম: উপজেলা মৎস্য কর্মকর্তা\nফোন: ০১৭৬৯-০৫৯১৪২\nঠিকানা: উপজেলা মৎস্য কর্মকর্তার কার্যালয়, চকরিয়া, কক্সবাজার।"
    }
}

# ড্রপডাউন মেনু ইন্টারফেস
selected_district = st.selectbox("১. আপনার জেলা নির্বাচন করুন:", ["-- জেলা নির্বাচন করুন --"] + list(officer_data.keys()))

if selected_district != "-- জেলা নির্বাচন করুন --":
    sub_districts = list(officer_data[selected_district].keys())
    selected_sub = st.selectbox("২. আপনার উপজেলা নির্বাচন করুন:", ["-- উপজেলা নির্বাচন করুন --"] + sub_districts)
    
    if selected_sub != "-- উপজেলা নির্বাচন করুন --":
        info = officer_data[selected_district][selected_sub]
        st.info(f"📋 **যোগাযোগের তথ্য:**\n\n{info}")
