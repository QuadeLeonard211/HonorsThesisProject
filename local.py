from ultralytics import YOLO
import cv2
import supervision as sv
import time
import numpy as np

class LocalDetector:
    def __init__(self, model_path):
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print(f"Model loaded. Available classes: {self.model.names}")
        print(f"Model task type: {self.model.task}")  # Add this to verify model type
        self.box_annotator = sv.BoxAnnotator(thickness=2)

    def initialize_webcam(self, camera_id: int = 0):
        """Initialize webcam capture."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        # Set camera properties to match Roboflow's settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame with YOLOv8.
        """
        # Ensure frame is in correct format
        if frame is None:
            return None
            
        # Print frame information
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        
        # Get YOLO predictions
        results = self.model(frame, conf=0.2, iou=0.45)  # Added IOU threshold
        
        # Debug information
        for r in results:
            if len(r.boxes) > 0:
                print("Detection found!")
                print(f"Boxes: {r.boxes}")
                print(f"Confidence: {r.boxes.conf}")
                print(f"Class IDs: {r.boxes.cls}")
        
        # Get the first result
        result = results[0]
        
        # Convert detections to supervision format
        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int)
        )
        
        # Draw the detections
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        
        # Add detection count overlay
        cv2.putText(
            annotated_frame,
            f"Detections: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return annotated_frame

    def run(self):
        """Run real-time detection on webcam feed."""
        try:
            cap = self.initialize_webcam()
            print("Starting detection... Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading from webcam")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                if annotated_frame is None:
                    continue
                
                # Show frame
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full error trace
            
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    MODEL_PATH = "nerf-strongarm/runs/detect/train/weights/best.pt"
    detector = LocalDetector(MODEL_PATH)
    detector.run()

if __name__ == "__main__":
    main()