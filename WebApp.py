import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="AI Colorization", layout="wide")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(
        "model/colorization_deploy_v2.prototxt",
        "model/colorization_release_v2.caffemodel"
    )
    pts = np.load("model/pts_in_hull.npy")

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

net = load_model()

# ---------------- Image Colorization ----------------
def colorize_image(img):
    img = np.array(img.convert("RGB"))
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))

    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, None], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    return (255 * colorized).astype("uint8")

# ---------------- Header ----------------
st.markdown("## 🎨 AI Grayscale Image & Video Colorization")
st.caption("Deep learning based colorization using CNNs and LAB color space")

# ---------------- Tabs (MAJOR UI CHANGE) ----------------
tab_img, tab_vid, tab_about = st.tabs(["🖼 Image", "🎬 Video", "ℹ About"])

# ================= IMAGE TAB =================
with tab_img:
    st.subheader("Image Colorization")

    uploaded = st.file_uploader("Upload a grayscale image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(img, use_column_width=True)

        if st.button("🎨 Colorize Image"):
            with st.spinner("Colorizing image..."):
                result = colorize_image(img)

            with col2:
                st.markdown("**Colorized Output**")
                st.image(result, use_column_width=True)

            st.download_button(
                "⬇ Download Colorized Image",
                data=Image.fromarray(result).tobytes(),
                file_name="colorized.png",
                mime="image/png"
            )
    else:
        st.info("Upload an image to begin.")

# ================= VIDEO TAB =================
with tab_vid:
    st.subheader("Video Colorization (Preview)")

    st.warning("This feature is computationally intensive.")
    st.info("Currently supports real-time preview only. Full video export is future scope.")

    uploaded_video = st.file_uploader("Upload a grayscale video")

    if uploaded_video and st.button("▶ Preview Colorized Video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            scaled = frame.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0].transpose((1, 2, 0))

            ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
            L = cv2.split(lab)[0]

            colorized = np.concatenate((L[:, :, None], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")

            stframe.image(colorized)

# ================= ABOUT TAB =================
with tab_about:
    st.markdown("""
### About the Project
This project uses a CNN-based pretrained colorization model to predict chrominance
values for grayscale images and videos using the LAB color space.

### Features
- Image colorization with comparison and download
- Video colorization with real-time preview
- Streamlit-based deployment

### Future Scope
- GPU acceleration
- Full video export
- Side-by-side video comparison
""")
