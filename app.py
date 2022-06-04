import pickle 
import numpy as np 
import streamlit as st
import cv2



global pca,pred,model,image
global result

global Count 
Count=0

#pca=pk.load(open('pca.pkl','rb'))
#model=pk.load(open('model.pkl','rb'))

pickle_pca = open("pca.pkl","rb")
pca=pickle.load(pickle_pca)

pickle_model = open("model.pkl","rb")
model=pickle.load(pickle_model)


st.subheader('Hand Written Digit Recognizer')

uploaded_file = st.file_uploader("Choose a image file", type=["jpg","png","jpeg"])
 


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # check if size is 28x28
    if (opencv_image.shape[0]==28)and(opencv_image.shape[1]==28):
        # Now do something with the image! For example, let's display it:
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        image=image.reshape(1,784)
        img_pca=pca.transform(image)
        
    else:
        resized_image = cv2.resize(opencv_image, (28, 28))
        image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        image=image.reshape(1,784)
        img_pca=pca.transform(image)
        
        
    st.image(uploaded_file,width=None)
col1,col2=st.columns(2)
with col1:
    if st.button('Click to Predict'):
        pred = model.predict(img_pca)
        result = pred[0]
        Count=1
        

with col2:
    if Count==1:
        st.write('Predicted Value = '+result)
    else:
        st.write('......')
   
   
    
   
     

