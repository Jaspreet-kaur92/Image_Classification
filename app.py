from flask import Flask, render_template, request
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import keras.utils as image

app = Flask(__name__)


model = load_model('data_model.h5')


def predict_image(img_path):
    
 
  test_im = image.load_img(path= img_path, target_size = (256, 256))
  
  test_i = image.img_to_array(test_im)
  test_image = np.expand_dims(test_i, axis = 0)
  result = model.predict(test_image)
  Bed, Chair, Sofa = result[0]

  if Bed==1.:
      result= 'Bed'
  elif Chair==1.:
      result = 'Chair'
  else:
      result = 'Sofa'
  return result

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # To get the image from post methd
        f = request.files['image_files']
        basepath = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)
       
        # Make prediction
        preds = predict_image(img_path)
        
    return render_template('home.html', prediction = preds, img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)