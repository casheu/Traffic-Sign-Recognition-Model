import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

# Load Sequential Model
model = load_model('model.h5')

def run():
    upload = st.file_uploader("Please upload an image", type=["jpg", "png"])
    
    if upload is not None:
        img =  Image.open(upload)
        resize_image = img.resize((32, 32))
        X = np.array(resize_image)
        X = X/255
        X_inf = np.expand_dims(X, axis=0)

        inf_pred = model.predict(X_inf)
        inf_class =np.argmax(inf_pred,axis=1)

        st.image(upload)
        st.write('## Prediction : ', inf_class[0])

        st.write('## Metadata : ')
        st.image('meta.png')

if __name__ == '__main__':
    run()
