from keras.models import load_model
from mtcnn import MTCNN
from PIL import Image


import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt


import cv2


detector = MTCNN()
emotion_dict = {
    0: 'Anger',
    1: 'Contempt', 
    2: 'Disgust',
    3: 'Fear',
    4: 'Happy',
    5: 'Sadness',
    6: 'Surprise'
}

# load the models
gender_model = load_model('./model/gender_model.h5')
emotion_model = load_model('./model/emotion_model.h5')

print('Model loaded successfully')


def pipeline_model(path, filename):
    
    img = cv2.imread(path)
    mt_res = detector.detect_faces(img)
    
    for i, face in enumerate(mt_res):
        
        x, y, width, height = face['box']
        center = [x+(width/2), y+(height/2)]
        max_border = max(width, height)
        
        # center alignment
        left = max(int(center[0]-(max_border/2)), 0)
        right = max(int(center[0]+(max_border/2)), 0)
        top = max(int(center[1]-(max_border/2)), 0)
        bottom = max(int(center[1]+(max_border/2)), 0)
        
        # crop the face
        cimg = img[top:top+max_border, 
                           left:left+max_border, :]
        center_img = np.array(Image.fromarray(cimg).resize([128, 128]))
        
        # gender prediction
        gender_pred = gender_model.predict(center_img.reshape(1,128,128,3))[0][0]
        
        # emotion prediction
        gray_img = np.array(Image.fromarray(cimg).resize([48, 48]))
        emotion_pred = emotion_model.predict(gray_img.reshape(1, 48, 48, 3))
        
        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        gender_text = 'Male' if gender_pred > 0.5 else 'Female'
        cv2.putText(img, 'Gender: {}'.format(gender_text), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
        cv2.putText(img, 'Emotion: {}'.format(emotion_dict[np.argmax(emotion_pred)]), (left, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
        
        cv2.imwrite('./static/predict/{}'.format(filename),img)
    