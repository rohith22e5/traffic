import cv2
import numpy as np
import os
import sys
import tensorflow as tf

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

trafficlabels={
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def main():

    model = tf.keras.models.load_model('88perfect.h5')
    image = convert_to_ppm('test3.jpg')
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
   

    image = np.expand_dims(image, axis=0) 
    result = model.predict(image)
    predicted_classes = np.argmax(result, axis=1)
    print(result)
    print(predicted_classes)
    print(trafficlabels[predicted_classes[0]])

def convert_to_ppm(image_path):
    # Read the input image

    image = cv2.imread(image_path)

   
    
    # Check if the image was successfully loaded
    if image is None:
        print("Error: Could not open or find the image.")
        return None

    # Convert the image to PPM format in memory
    _, ppm_data = cv2.imencode('.ppm', image, [cv2.IMWRITE_PXM_BINARY, 1])

    # Convert PPM data to bytes array
    ppm_bytes = np.array(ppm_data).tobytes()

    return ppm_bytes



if __name__=="__main__":
    main()
