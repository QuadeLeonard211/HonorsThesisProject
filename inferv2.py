import cv2
import base64
import numpy as np
import requests
import time
from typing import Optional, Tuple

class RoboflowWebcam:
    def __init__(self, api_key: str, model_name: str, size: int = 157):
        self.api_key = api_key
        self.model_name = model_name
        self.size = size
        self.upload_url = f"https://detect.roboflow.com/{model_name}?access_token={api_key}&format=image&stroke=5"
        self.video = None

    def initialize_webcam(self, camera_id: int = 0) -> bool:
        """Initialize the webcam with error handling."""
        try:
            self.video = cv2.VideoCapture(camera_id)
            if not self.video.isOpened():
                print("Error: Could not open webcam")
                return False
            return True
        except Exception as e:
            print(f"Error initializing webcam: {str(e)}")
            return False

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from the webcam with validation."""
        if self.video is None:
            return False, None
        
        ret, frame = self.video.read()
        if not ret or frame is None:
            return False, None
            
        return True, frame

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        height, width, _ = img.shape
        scale = self.size / max(height, width)
        return cv2.resize(img, (round(scale * width), round(scale * height)))

    def infer(self) -> Optional[np.ndarray]:
        """Get inference from Roboflow API with error handling."""
        try:
            # Capture and validate frame
            ret, img = self.capture_frame()
            if not ret or img is None:
                print("Failed to capture frame")
                return None

            # Resize image
            img = self.resize_image(img)

            # Encode image to base64
            retval, buffer = cv2.imencode('.jpg', img)
            if not retval:
                print("Failed to encode image")
                return None
            img_str = base64.b64encode(buffer)

            # Make API request with timeout
            response = requests.post(
                self.upload_url,
                data=img_str,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10  # 10 second timeout
            )
            
            if response.status_code != 200:
                print(f"API request failed with status code: {response.status_code}")
                return img  # Return original image if API fails
                
            # Parse result image
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Failed to decode response image")
                return img  # Return original image if decoding fails
                
            return image

        except requests.exceptions.Timeout:
            print("API request timed out")
            return img
        except requests.exceptions.RequestException as e:
            print(f"API request error: {str(e)}")
            return img
        except Exception as e:
            print(f"Unexpected error during inference: {str(e)}")
            return img

    def run(self):
        """Main loop for capturing and processing video."""
        if not self.initialize_webcam():
            return

        try:
            while True:
                # Check for 'q' keypress
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Get inference result
                image = self.infer()
                
                # Display the frame if valid
                if image is not None:
                    cv2.imshow('Roboflow YOLO Detection', image)
                
                # Add a small delay to reduce CPU usage
                time.sleep(0.01)

        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        
        finally:
            # Clean up resources
            if self.video is not None:
                self.video.release()
            cv2.destroyAllWindows()

def main():
    # Your API credentials
    ROBOFLOW_API_KEY = "h04D2imkPyxDevx6BIFg"
    ROBOFLOW_MODEL = "nerf-strongarm"
    ROBOFLOW_SIZE = 416

    # Initialize and run
    detector = RoboflowWebcam(ROBOFLOW_API_KEY, ROBOFLOW_MODEL, ROBOFLOW_SIZE)
    detector.run()

if __name__ == "__main__":
    main()