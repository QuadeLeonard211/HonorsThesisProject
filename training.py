from ultralytics import YOLO
import torch

def train_yolo(data_yaml_path: str, epochs: int = 50):
    """
    Train a YOLOv8 model on your custom dataset.
    
    Args:
        data_yaml_path: Path to your data.yaml file from Roboflow
        epochs: Number of training epochs
    """
    try:
        # Initialize a new YOLOv8 model
        model = YOLO('yolov8n.pt')  # Start with pre-trained nano model
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            patience=10,  # Early stopping patience
            save=True,  # Save best model
            device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        )
        
        # Save the trained model
        model.save('strongarm_detector.pt')
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    # Path to your data.yaml file from the Roboflow download
    DATA_YAML_PATH = "nerf-strongarm/data.yaml"
    
    # Train the model
    train_yolo(DATA_YAML_PATH, epochs=50)

if __name__ == "__main__":
    main()