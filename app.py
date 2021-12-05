import numpy as np
import sys
import os
import glob
import re


 
from tensorflow.keras.applications.imagenet_utils import preprocess_input ,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

from flask import Flask,render_template,redirect,url_for,request

  # Flask app
app = Flask(__name__)
model_path='vgg19.h5'



# Load model
model = load_model(model_path)
model.make_predict_function() 


# preprocessing functio
def model_predict(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
        
    
      # preprocessing to the image
    x=image.img_to_array(img)
   
    x=np.expand_dims(x,axis=0)
   
    x=preprocess_input(x)
   
    preds=model.predict(x)
    return preds





@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        # GET file from post
        f=request.files['file']
        
        #save the path to upload
        basepath = os.path.dirname(__file__)
        
        file_path =os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
      # we are going to make prediction
      
        pred=model_predict(file_path,model)
        
        
        pred_class =decode_predictions(pred, top=1)
        result =str(pred_class[0][0][1])
        return result
    return None
        


if __name__ =='__main__':
    app.run(debug=True)


