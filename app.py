import logging

import streamlit as st

from inference import InvalidImageError, decode_image, load_model, predict_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_model():
    return load_model()


st.set_page_config(page_title="Car Recognition", layout="centered")
st.markdown(
    """
    <style>
    .block-container {max-width: 880px; padding-top: 3rem; padding-bottom: 4rem;}
    [data-testid="stFileUploader"] {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Car Recognition")
st.write("Upload a photo to find the car model and see how confident the model is.")

uploaded_image = st.file_uploader(
    "Choose a car image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, PNG, and WebP.",
)

if uploaded_image is not None:
    image_column, result_column = st.columns([1, 1], gap="large")

    try:
        image = decode_image(uploaded_image.getvalue())

        with image_column:
            st.image(image, caption="Your uploaded image", width="stretch")

        with result_column:
            with st.spinner("Identifying the car..."):
                model, class_names = get_model()
                predicted_name, confidence = predict_image(image, model, class_names)

            with st.container(border=True):
                st.caption("Predicted model")
                st.subheader(predicted_name)
                st.metric("Confidence", f"{confidence:.1%}")
    except InvalidImageError as error:
        with result_column:
            st.error(str(error))
    except FileNotFoundError:
        logger.exception("The trained model checkpoint could not be found.")
        with result_column:
            st.error("The trained model is missing. Please contact the app maintainer.")
    except Exception:
        logger.exception("Car image prediction failed.")
        with result_column:
            st.error("We couldn't analyze this image. Please try another image.")
else:
    st.info("Upload a clear photo of a car to get started.")
