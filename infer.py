# h04D2imkPyxDevx6BIFg

ROBOFLOW_API_KEY = "h04D2imkPyxDevx6BIFg"
ROBOFLOW_MODEL = "nerf-strongarm" # eg xx-xxxx--#
ROBOFLOW_SIZE = 416

import cv2
import base64
import numpy as np
import requests

upload_url = "".join([
    "https://detect.roboflow.com/", # http://127.0.0.1:9001
    ROBOFLOW_MODEL,
    "?access_token=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

video = cv2.VideoCapture(0)

# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    if (not image): # image was not created 
        print(image)
    # And display the inference results
    cv2.imshow('image', image)

# Release resources when finished
video.release()
cv2.destroyAllWindows()