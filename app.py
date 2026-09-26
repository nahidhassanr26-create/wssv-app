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

# পেজ সেটিংস
st.set_page_config(page_title="WSSV ডিটেকশন ও পরামর্শ কেন্দ্র", page_icon="🦐", layout="centered")

@st.cache_resource
def load_model():
    return YOLO('best.tflite')

try:
    model = load_model()
except Exception as e:
    st.error("মডেল লোড হতে সমস্যা হচ্ছে।")

st.title("🦐 WSSV (হোয়াইট স্পট) ডিটেকশন ও পরামর্শ কেন্দ্র")
st.write("আপনার চিংড়ির ছবি আপলোড করে বা সরাসরি ক্যামেরা দিয়ে তুলে ২৪/৭ পরীক্ষা করুন।")

uploaded_file = st.file_uploader("চিংড়ির ছবি নির্বাচন করুন...", type=["jpg", "jpeg", "png"])
cam_file = st.camera_input("অথবা সরাসরি ছবি তুলুন")

image = None
if uploaded_file is not None: image = Image.open(uploaded_file)
elif cam_file is not None: image = Image.open(cam_file)

# পিডিএফ জেনারেটর ফাংশন
def generate_pdf(status, advice, spot_cnt, severity, conf_pct, img_array):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    # কাস্টম স্টাইল (ইংরেজি ফন্ট ব্যবহার করা হয়েছে পিডিএফ ফ্রেন্ডলি করার জন্য)
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
    
    # ইমেজ প্রসেস ও পিডিএফে যুক্ত করা
    rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='JPEG', quality=80)
    img_buffer.seek(0)
    
    rl_img = RLImage(img_buffer, width=300, height=300)
    story.append(rl_img)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<b>Actionable Advice / Recommendations:</b>", ParagraphStyle('Sub', parent=styles['Heading2'], fontSize=14, spaceAfter=8)))
    # বাংলা পরামর্শ পিডিএফে এড়াতে ইংরেজিতে সুন্দর করে ক্লিন গাইডলাইন দেওয়া হলো
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
    
    # সেভ স্টেট তৈরি করা যাতে বাটন ক্লিকে ইমেজ হারিয়ে না যায়
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
                img_cv = r.plot()
        
        output_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        st.subheader("📊 পরীক্ষার ফলাফল:")
        st.image(output_image, use_container_width=True)
        
        conf_percentage = int(max_conf * 100)
        
        if spot_count > 0 and max_conf >= 0.05:
            severity = "কম ঝুঁকি (Low)" if spot_count <= 2 else "মাঝারি ঝুঁকি (Medium)" if spot_count <= 5 else "উচ্চ ঝুঁকি (High Risk)"
            status_title = "WSSV Risk Detected"
            
            st.error(f"🔴 WSSV (হোয়াইট স্পট) আক্রান্ত হওয়ার ঝুঁকি (Risk) পাওয়া গেছে!")
            st.write(f"* **মডেল সর্বোচ্চ কনফিডেন্স:** {conf_percentage}%")
            st.write(f"* **চিহ্নিত লক্ষণের সংখ্যা:** {spot_count} টি")
            st.warning(f"🚨 ঝুঁকির মাত্রা: {severity}")
            
            advice_html = """
            **🚨 তাত্ক্ষণিক জরুরি পরামর্শ (Actionable Advice):**
            1. **পানি বিনিময় বন্ধ করুন:** এই পুকুর বা ঘেরের পানি অন্য কোথাও ছড়াতে দেবেন না।
            2. **Emergency Harvest:** চিংড়ি বিক্রির উপযোগী সাইজের কাছাকাছি হলে দ্রুত ধরে ফেলুন, নয়তো ব্যাপক মড়ক হতে পারে।
            3. **বিশেষজ্ঞের সহায়তা:** অবিলম্বে আপনার নিকটস্থ উপজেলা মৎস্য কর্মকর্তা বা অ্যাকুয়াকালচার ল্যাবের সাথে যোগাযোগ করুন।
            """
            st.markdown(advice_html)
        else:
            severity = "No Risk"
            status_title = "Healthy Shrimp / No Sign of WSSV"
            st.success("🟢 কোনো উল্লেখযোগ্য WSSV ঝুঁকি পাওয়া যায়নি (সুস্থ চিংড়ি)")
            st.markdown("""
            **📋 সাধারণ পরামর্শ:**
            * খামারের সাধারণ বায়োসিকিউরিটি বজায় রাখুন।
            * নিয়মিত পানির গুণাগুণ পরীক্ষা করুন।
            """)
            advice_html = "Maintain strict biosecurity and monitor water parameters."
            
        # পিডিএফ জেনারেশন বাটন যুক্ত করা
        pdf_data = generate_pdf(status_title, advice_html, spot_count, severity, conf_percentage, img_cv)
        
        st.write("---")
        st.download_button(
            label="📥 পরীক্ষার পিডিএফ রিপোর্ট ডাউনলোড করুন",
            data=pdf_data,
            file_name=f"WSSV_Report_{datetime.date.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
