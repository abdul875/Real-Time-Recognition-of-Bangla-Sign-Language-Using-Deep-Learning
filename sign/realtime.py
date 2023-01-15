import cv2
import os
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from PIL import ImageFont, ImageDraw, Image
import time

frequency = 1500
duration = 500

font_scale = 1.5
fontface = cv2.FONT_HERSHEY_PLAIN
Xception = tf.keras.models.load_model('sign/modelG_Xception.h5')


#status = ['Forgive','Telephone','Name','Deaf','Closed','Age','Mother','Support','Alert','Salam','0','1','2','3','4','5','6','7','8','9' ]
status= ['ক্ষ্মা','টেলিফোন','নাম','বধির','বন্ধ','বয়স','মা','সমর্থন','সতর্ক','সালাম','০','১','২','৩','৪','৫','৬','৭','৮','৯']

def image_pred(image):
    current_status = ""

    ## Make canvas and set the color
    img = np.zeros((200,400,3),np.uint8)
    b,g,r,a = 0,255,0,0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_face = resize(gray, (224, 224, 3))
    final = np.expand_dims(resized_face, axis=0)
    pred = Xception.predict(final)
    current_status = status[np.argmax(pred)]
    print(current_status)

    fontpath = "E:\\Ewu\\13th semester\\B_Sign_L\\sign\\kalpurush.ttf" 
    font = ImageFont.truetype(fontpath, 50)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((50, 80), current_status , font = font, fill = (b, g, r, a))
    img = np.array(img_pil)


    #cv2.putText(img, current_status, (100, 150),
                 #fontface, 3, (34, 153, 84), 2, cv2.LINE_4)


    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()
