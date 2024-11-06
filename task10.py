import cv2
from picamera2 import Picamera2
import numpy as np
import time
import os
from datetime import datetime
import torch
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import transforms
from PIL import Image

class CameraApp:
    def __init__(self):
        # Initialize camera
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (1280, 720)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.configure("preview")
        self.picam2.start()

        # Initilize model
        # Define the model (same architecture as during training)
        self.model = mobilenet_v3_small(weights=None, num_classes=2)

        # Load the state_dict (weights) from the saved model
        self.model.load_state_dict(torch.load('best_model_ce.pth'))

        # Now set the model to evaluation mode
        self.model.eval()

        self.faceID = ""

        # Define the transform to resize the image to 64x64 and normalize
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust based on model training
        ])

        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_mode = False  # Toggle for face cropping mode

        # Initialize recording variables
        self.is_recording = False
        self.output_video = None
        self.face_video = None
        self.start_time = None

        # Create output directories
        self.image_dir = "captured_images"
        self.video_dir = "captured_videos"
        self.face_dir = "captured_faces"
        for directory in [self.image_dir, self.video_dir, self.face_dir]:
            os.makedirs(directory, exist_ok=True)

        # Window names
        self.main_window = "Camera App"
        self.face_window = "Face View"
        cv2.namedWindow(self.main_window)
        cv2.namedWindow(self.face_window)

    def detect_face(self, frame):
        """Detect and return the largest face in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(150, 150)
        )
        
        # Get the largest face
        if len(faces) > 0:
            # Find the face with the largest area
            areas = [w * h for (x, y, w, h) in faces]
            largest_face = faces[np.argmax(areas)]
            return largest_face
        return None

    def generate_filename(self, file_type, face=False):
        """Generate unique filename based on timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if file_type == "image":
            base_dir = self.face_dir if face else self.image_dir
            return os.path.join(base_dir, f"{'face' if face else 'image'}_{timestamp}.jpg")
        else:  # video
            base_dir = self.face_dir if face else self.video_dir
            return os.path.join(base_dir, f"{'face' if face else 'video'}_{timestamp}.mp4")

    def draw_ui(self, frame):
        """Draw UI elements on the frame"""
        # Add recording indicator
        if self.is_recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            elapsed_time = int(time.time() - self.start_time)
            duration = f"REC {elapsed_time//60:02d}:{elapsed_time%60:02d}"
            cv2.putText(frame, duration, (50, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add face mode indicator
        face_status = "Face Mode: ON" if self.face_mode else "Face Mode: OFF"
        cv2.putText(frame, face_status, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add instructions
        instructions = [
            "Press 'r' to start/stop recording",
            "Press 'c' to capture image",
            "Press 'f' to toggle face mode",
            "Press 'q' to quit"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 20 - (i * 30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

  

    def capture_image(self, frame, face_crop=None):
        """Capture images - both full frame and face crop if available"""
        # Save full frame
        # filename = self.generate_filename("image")
        # cv2.imwrite(filename, frame)
        # print(f"Image captured: {filename}")

        # Save face crop if available
        if face_crop is not None:
            face_filename = self.generate_filename("image", face=True)
            face_crop_64 = cv2.resize(face_crop, (64, 64))
            # cv2.imwrite(face_filename, face_crop_64)
            self.checkModel(face_crop_64)
            print(f"Face captured: {face_filename}")

    def checkModel(self, face_crop_64):
        # Apply the transformation (resize to 64x64 and normalize)
        # Convert the face crop from BGR (OpenCV format) to RGB (PIL format)
        face_crop_rgb = cv2.cvtColor(face_crop_64, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for compatibility with transforms
        pil_face = Image.fromarray(face_crop_rgb)

        # Apply the transformations (resize to 64x64 and normalize)
        input_tensor = self.transform(pil_face).unsqueeze(0)  # Add batch dimension

        # Make the prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted_class = torch.max(output, 1)

        # Output the prediction
        if predicted_class.item() == 0:
            self.faceID= "Hayden"
        else:
            self.faceID= "Conor"


    def run(self):
        """Main application loop"""
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                display_frame = frame.copy()
                face_crop = None
                
                # Detect face if face mode is on
                if self.face_mode:
                    face_rect = self.detect_face(frame)
                    if face_rect is not None:
                        x, y, w, h = face_rect
                        # Draw rectangle on display frame
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(display_frame, self.faceID,(x-20, y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2 )
                        # Crop face
                        face_crop = frame[y:y+h, x:x+w]
                        # Show face crop
                        if face_crop is not None:
                            cv2.imshow(self.face_window, face_crop)
                            
                            # Initialize face video writer if recording and not yet initialized
                            if self.is_recording and self.face_video is None:
                                h, w = face_crop.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                self.face_video = cv2.VideoWriter(
                                    self.face_filename, fourcc, 3, (w, h))
                
                # Draw UI on display frame
                display_frame = self.draw_ui(display_frame)
                
                # Show main frame
                cv2.imshow(self.main_window, display_frame)
                
                # Record frames
                if self.is_recording:
                    self.output_video.write(frame)  # Write full frame without UI
                    if self.face_video is not None and face_crop is not None:
                        self.face_video.write(face_crop)  # Write face crop
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # if self.is_recording:
                    #     self.stop_recording()
                    break
                # elif key == ord('r'):
                #     if not self.is_recording:
                #         self.start_recording()
                #     else:
                #         self.stop_recording()
                elif key == ord('c'):
                    self.capture_image(frame, face_crop)
                elif key == ord('f'):
                    self.face_mode = not self.face_mode
                    if not self.face_mode:
                        cv2.destroyWindow(self.face_window)

        finally:
            # Cleanup
            # if self.is_recording:
            #     self.stop_recording()
            self.picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CameraApp()
    app.run()