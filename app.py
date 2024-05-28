import os
import shutil
from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import zipfile

app = Flask(__name__)

# base directory 
base_dir = os.path.dirname(os.path.abspath(__file__))

# YOLO model check 
weights_path = os.path.join(base_dir, "yolov3.weights")
config_path = os.path.join(base_dir, "yolov3.cfg")
names_path = os.path.join(base_dir, "yolov3.txt")

# YOLO model load
net = cv2.dnn.readNet(weights_path, config_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# class lable 
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

#extensions which can be processed
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# index.html render
@app.route('/')
def home():
    return render_template('index.html')

# post method
@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    files = request.files.getlist('files')
    resize_width = 0
    resize_height = 0
    if (request.form.get('width')):
        resize_width = int(request.form.get('width', 416))  
    if (request.form.get('height')):
        resize_height = int(request.form.get('height', 416))  # use 416 as default

    
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    
    #output directory check
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    for file in files:
        filename = secure_filename(file.filename)
        if is_image_file(filename):
            img = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            # image classification
            if img is not None:
                height, width, channels = img.shape
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        class_dir = os.path.join(output_dir, label)
                        if not os.path.exists(class_dir):
                            os.makedirs(class_dir)
                        
                        
                        img_path = os.path.join(class_dir, filename)
                        # app.logger.info('Original size: %s', img.shape)
                        
                        # check if user input width and height
                        if resize_height != 0 and resize_width != 0:
                            
                            #img resize 
                            resized_img = cv2.resize(img, (resize_width, resize_height))
                            cv2.imwrite(img_path, resized_img)
                        else :
                            cv2.imwrite(img_path, img)
                        
                        #app.logger.info('Resized : %s', resized_img.shape)
                        
                    
                    
    #upload classified images as a zip file
    zip_path = os.path.join(base_dir, 'classified.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, output_dir))

    shutil.rmtree(output_dir)

    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
