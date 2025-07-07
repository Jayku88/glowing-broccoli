# import tensorflow as tf
# model = tf.keras.models.load_model('newmodel1_feb_3.h5')
# import streamlit as st
# st.title("Skin Disease Classification")
# st.write("This is a simple image classification web app to predict skin diseases into (0: Seborrheic keratosis, 1: Squamous cell carcinoma, 2: Vascular lesion)")
# #model.summary(print_fn=lambda x: st.text(x))
# file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
# #import cv2
# from PIL import Image, ImageOps
# import numpy as np


# def import_and_predict(image_data, model):
#     size = (224,224)
#     #image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#     image = ImageOps.fit(image_data, size)
#     image = np.asarray(image)
#     img = np.expand_dims(image, axis=0)
#     prediction = model.predict(img)

#     return prediction


# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)
#     st.image(image, use_container_width=True)
#     prediction = import_and_predict(image, model)

#     if np.argmax(prediction) == 0:
#         st.write("""
#                  # Seborrheic keratosis
#                  """
#                  )
#     elif np.argmax(prediction) == 1:
#         st.write("""
#                         # Squamous cell carcinoma
#                         """
#                  )
#     else:
#         st.write("""
#                                # Vascular lesion
#                                """
#                  )
#     st.write('Probability predicted for Seborrheic keratosis', prediction[0][0])
#     st.write('Probability predicted for Squamous cell carcinoma',prediction[0][1])
#     st.write('Probability predicted for Vacular lesion',prediction[0][2])

import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Load model
model = tf.keras.models.load_model('attunet8c3dgpatch.h5')  # e.g., "road_segmentation_model.h5"

# UI
st.title("Road Extraction using Semantic Segmentation")
st.write("Upload a satellite image and the model will extract road regions.")

# File uploader
file = st.file_uploader("Please upload a satellite image", type=["jpg", "png", "jpeg"])

# Image preprocessing + prediction
def preprocess_image(image_data, target_size=(512, 512)):
    image = ImageOps.fit(image_data, target_size, method=Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0  # Normalize
    #image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# def postprocess_mask(mask):
#     # Assuming mask is (1, H, W, 1) with values 0-1
#     mask = tf.squeeze(mask, axis=0)     # (H, W, 1)
#     mask = tf.squeeze(mask, axis=-1)    # (H, W)
#     mask = tf.cast(mask > 0.5, tf.uint8) * 255  # Threshold + scale to 0/255
#     return mask.numpy()

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    input_image = preprocess_image(image)
    prediction = model.predict(input_image)
    predictions = (predictions > 0.5).astype(np.uint8)

    mask = postprocess_mask(prediction)

    st.image(mask, caption='Predicted Road Mask', use_container_width=True, clamp=True)

else:
    st.info("Please upload an image.")

