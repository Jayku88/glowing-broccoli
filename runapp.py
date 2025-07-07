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
model = tf.keras.models.load_model('attunet8c3dgpatch.h5',compile=False)  # e.g., "road_segmentation_model.h5"

# UI
st.title("Road Extraction using Semantic Segmentation")
st.write("Upload a satellite image and the model will extract road regions.")

# File uploader
file = st.file_uploader("Please upload a satellite image", type=["jpg", "png", "jpeg"])

# Image preprocessing + prediction
# def preprocess_image(image_data, target_size=(512, 512)):
#     image_array = np.array(image_data)           # Convert to NumPy array first
#     image_array = image_array / 255.0 
#     return image_array

def preprocess_image(image_data, target_size=(512, 512)):
    # Ensure image is RGB
    image_data = image_data.convert('RGB')
    
    # Resize image
    image_data = image_data.resize(target_size)
    
    # Convert to NumPy array and normalize
    image_array = np.array(image_data).astype(np.float32) / 255.0  # (512, 512, 3)
    image_array = np.expand_dims(image_array, 0)
    # image_array = np.array(image_data / 255.0 )
    # image_array = np.expand_dims(image_array, 0)
    return image_array

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    input_image = preprocess_image(image)
    
    # prediction = model.predict(input_image)
    # predictions = (prediction > 0.1).astype(np.uint8)

    prediction = model.predict(input_image)
    #predictions = (prediction > 0.5).astype(np.uint8)
    
    print(prediction.shape)
    mask = prediction.squeeze()          # shape: (512, 512)
    print(mask.shape)
    mask = (mask * 255).astype(np.uint8)  # scale binary to [0, 255]
    mask_image = Image.fromarray(mask)

    #mask = postprocess_mask(prediction)

    #st.image(predictions, caption='Predicted Road Mask', use_container_width=True, clamp=True)
    st.image(predictions, caption='Predicted Road Mask', use_container_width=True)

else:
    st.info("Please upload an image.")

