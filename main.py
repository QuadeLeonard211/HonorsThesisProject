from ultralytics import YOLO
from roboflow import Roboflow

# Load into the Roboflow database
rf = Roboflow(api_key="h04D2imkPyxDevx6BIFg")
project = rf.workspace("quade-leonard-qmmok").project("nerf-strongarm")
version = project.version(1)
dataset = version.download("yolov8")

# Load the Model
# model = project.version(dataset.version).model

