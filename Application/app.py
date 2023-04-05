from flask import Flask,render_template,request,Response
from pipeline.pipeline import PreprocessPipeline
from PIL import Image
import cv2 
import io
from  flaskwebgui import FlaskUI

app = Flask(__name__)
pipeline = PreprocessPipeline('Application\pipeline\models\FIRSTMODEL.h5')
ui = FlaskUI(app,width=500, height=500)




@app.route('/')
def upload_form():
    return render_template('upload.html')



@app.route('/ImageUploader',methods =['GET','POST'])
def ImageUploader():
    image = request.files['image']
    image = Image.open(image)

    image_output = pipeline.process(image)
    image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB)

    _, img_buffer = cv2.imencode('.jpg', image_output)
    img_buffer = io.BytesIO(img_buffer)
    img_buffer.seek(0)

    return Response(img_buffer, mimetype='image/jpeg')






if __name__ == '__main__':
    app.run()