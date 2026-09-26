import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ১. পেজ সেটিংস
st.set_page_config(page_title="WSSV ডিটেকশন ও পরামর্শ কেন্দ্র", page_icon="🦐", layout="centered")

# ২. মডেল লোড করা (ক্যাশ মেমোরি ব্যবহার করে যাতে ফাস্ট লোড হয়)
@st.cache_resource
def load_model():
    return YOLO('best.tflite')

try:
    model = load_model()
except Exception as e:
    st.error("মডেল লোដ করতে সমস্যা হচ্ছে। 'best.tflite' ফাইলটি সঠিক জায়গায় আছে কিনা চেক করুন।")

# অ্যাপের শিরোনাম
st.title("🦐 WSSV (হোয়াইট স্পট) ডিটেকশন ও পরামর্শ কেন্দ্র")
st.write("আপনার চিংড়ির ছবি আপলোড করে বা সরাসরি ক্যামেরা দিয়ে তুলে ২৪/৭ পরীক্ষা করুন।")

# ৩. ইনপুট অপশন
uploaded_file = st.file_uploader("চিংড়ির ছবি নির্বাচন করুন...", type=["jpg", "jpeg", "png"])
cam_file = st.camera_input("অথবা সরাসরি ছবি তুলুন")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif cam_file is not None:
    image = Image.open(cam_file)

# ৪. প্রসেসিং ও ফিক্সড ৫% রিস্ক লজিক
if image is not None:
    st.image(image, caption='আপনার দেওয়া ছবি', use_container_width=True)
    
    if st.button("পরীক্ষা করুন (Analyze)", type="primary"):
        with st.spinner('মডেল পরীক্ষা করছে... অনুগ্রহ করে অপেক্ষা করুন...'):
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
            
            # ফলাফল প্রদর্শন
            st.subheader("📊 পরীক্ষার ফলাফল:")
            st.image(output_image, caption='চিহ্নিত ফলাফল (Detection Result)', use_container_width=True)
            
            # কাস্টম রিস্ক অ্যালার্ট
            if spot_count > 0 and max_conf >= 0.05:
                severity = "কম ঝুঁকি (Low)" if spot_count <= 2 else "মাঝারি ঝুঁকি (Medium)" if spot_count <= 5 else "উচ্চ ঝুঁকি (High Risk)"
                
                st.error(f"🔴 WSSV (হোয়াইট স্পট) আক্রান্ত হওয়ার ঝুঁকি (Risk) পাওয়া গেছে!")
                st.write(f"* **মডেল সর্বোচ্চ কনফিডেন্স:** {int(max_conf*100)}%")
                st.write(f"* **চিহ্নিত লক্ষণের সংখ্যা:** {spot_count} টি")
                st.warning(f"🚨 ঝুঁকির মাত্রা: {severity}")
                
                st.markdown("""
                **🚨 তাত্ক্ষণিক জরুরি পরামর্শ (Actionable Advice):**
                ১. **पानी বিনিময় বন্ধ করুন:** এই পুকুর বা ঘেরের পানি অন্য কোথাও ছড়াতে দেবেন না।
                ২. **Emergency Harvest:** চিংড়ি বিক্রির উপযোগী সাইজের কাছাকাছি হলে দ্রুত ধরে ফেলুন, নয়তো ব্যাপক মড়ক হতে পারে।
                ৩. **বিশেষজ্ঞের সহায়তা:** অবিলম্বে আপনার নিকটস্থ উপজেলা মৎস্য কর্মকর্তা বা অ্যাকুয়াকালচার ল্যাবের সাথে যোগাযোগ করুন।
                """)
            else:
                st.success("🟢 কোনো উল্লেখযোগ্য WSSV ঝুঁকি পাওয়া যায়নি (সুস্থ চিংড়ি)")
                st.markdown("""
                **📋 সাধারণ পরামর্শ:**
                * খামারের সাধারণ বায়োসিকিউরিটি বজায় রাখুন।
                * নিয়মিত পানির গুণাগুণ পরীক্ষা করুন।
                """)
