import streamlit as st
import tensorflow as tf
from PIL import Image
import time
import qrcode
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import base64
from utils import load_model, predict

# Page config
st.set_page_config(page_title="Plant Disease Detection App", page_icon="app-images/logo-02.png")
st.title("🌿 Plant Disease Detection")
st.image("app-images/logo-01.png")
st.write("Upload a leaf image to detect plant disease and get treatment recommendations.")

# Load the TFLite model
interpreter = load_model(model_path="model/plant_model_5_class.tflite")
class_names = ['Healthy', 'Powdery', 'Rust', 'Slug', 'Spot']

# Recommendations dictionary with URLs
recommendations = {
    "Healthy": {
        "Organic": {
            "text": "✅ Your plant looks healthy!\n- Continue regular watering\n- Use compost and neem spray weekly\n- Maintain airflow and hygiene",
            "url": "https://example.com/organic-plant-tonic"
        },
        "Chemical": {
            "text": "✅ No issues detected!\n- Maintain preventive schedule using balanced NPK fertilizers\n- Apply mild fungicide every 30 days if needed",
            "url": "https://example.com/npk-fertilizer"
        }
    },
    "Powdery": {
        "Organic": {
            "text": "🦠 **Powdery Mildew - Organic Treatment**\n- Spray neem oil weekly\n- Use baking soda + water + liquid soap\n- Improve plant spacing",
            "url": "https://www.amazon.in/dp/B07KQJP5WV"
        },
        "Chemical": {
            "text": "🦠 **Powdery Mildew - Chemical Treatment**\n- Use sulfur-based fungicide like **Thiovit Jet**\n- Apply potassium bicarbonate like **MilStop**",
            "url": "https://www.amazon.in/dp/B07NQW7ZJ5"
        }
    },
    "Rust": {
        "Organic": {
            "text": "🍂 **Rust - Organic Treatment**\n- Remove infected leaves\n- Spray compost tea or neem oil\n- Avoid overhead watering",
            "url": "https://www.amazon.in/dp/B08X17LMQ6"
        },
        "Chemical": {
            "text": "🍂 **Rust - Chemical Treatment**\n- Apply **Indofil M-45** (Mancozeb)\n- Use **Kavach** (Chlorothalonil) every 7-10 days",
            "url": "https://www.amazon.in/dp/B094DKXXSM"
        }
    },
    "Slug": {
        "Organic": {
            "text": "🐌 **Slug Damage - Organic Treatment**\n- Use copper tape around pots\n- Sprinkle diatomaceous earth\n- Set beer traps at night",
            "url": "https://www.amazon.in/dp/B08XZNXLSZ"
        },
        "Chemical": {
            "text": "🐌 **Slug Damage - Chemical Treatment**\n- Apply **Sluggo** (Ferric phosphate-based bait)\n- Place pellets in moist shaded areas",
            "url": "https://www.amazon.in/dp/B000BX4GXS"
        }
    },
    "Spot": {
        "Organic": {
            "text": "🔴 **Leaf Spot - Organic Treatment**\n- Spray neem oil or compost tea\n- Use diluted hydrogen peroxide (5%)\n- Remove infected foliage",
            "url": "https://www.amazon.in/dp/B07KQJP5WV"
        },
        "Chemical": {
            "text": "🔴 **Leaf Spot - Chemical Treatment**\n- Use **Blitox 50** (copper-based fungicide)\n- Apply **Zineb** every 10 days during early stages",
            "url": "https://www.amazon.in/dp/B07D1BL6JH"
        }
    }
}

# Upload section
uploaded_file = st.file_uploader("📤 Upload an image of the leaf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    treatment_type = st.radio("🧪 Choose Treatment Type:", ["Organic", "Chemical"])

    if st.button("🔍 Predict"):
        start_time = time.time()
        predicted_class_name, probability = predict(image, class_names, interpreter)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        st.success(f"🌱 Predicted: **{predicted_class_name}** with **{probability:.2f}** confidence in {inference_time:.2f} ms")

        # Fetch treatment and product
        result = recommendations.get(predicted_class_name, {}).get(treatment_type)

        if result:
            st.markdown("### 💡 Recommendation:")
            st.write(result["text"])

            # Show QR code
            st.markdown("#### 📱 Product Purchase Link:")
            qr = qrcode.make(result["url"])
            qr_buf = BytesIO()
            qr.save(qr_buf)
            st.image(qr_buf.getvalue(), width=150)

            # Create downloadable PDF
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.setFont("Helvetica", 14)
            c.drawString(50, 750, f"Plant Disease: {predicted_class_name}")
            c.drawString(50, 730, f"Treatment Type: {treatment_type}")
            c.drawString(50, 710, "Recommendation:")
            text_lines = result["text"].split("\n")
            y = 690
            for line in text_lines:
                c.drawString(60, y, line)
                y -= 20
            c.drawString(50, y - 10, f"Product URL: {result['url']}")
            c.save()
            pdf_buffer.seek(0)

            # Show download link
            b64 = base64.b64encode(pdf_buffer.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="plant_disease_report.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.warning("No treatment recommendation found for this disease and option.")
