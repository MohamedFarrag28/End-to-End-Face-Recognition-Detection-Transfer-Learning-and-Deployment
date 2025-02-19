import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, model_weights_path=r"..\models\opencv_face_detector", confidence_threshold=0.5):
        """
        Initializes the FaceDetector with OpenCV's DNN model.

        Args:
            model_path (str): Path to the directory containing the model files.
            confidence_threshold (float): Minimum confidence for detecting a face.
        """
        self.confidence_threshold = confidence_threshold
        self.prototxt = os.path.join(model_weights_path, "deploy.prototxt")
        self.caffemodel = os.path.join(model_weights_path, "res10_300x300_ssd_iter_140000.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)


    def detect_faces(self, image):
        """
        Detects faces in an image using OpenCV DNN face detector.

        Args:
            image (numpy.ndarray): Image as a NumPy array.

        Returns:
            List of bounding boxes for detected faces [(x1, y1, x2, y2), ...].
        """
        (h, w) = image.shape[:2]

        # Prepare input blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        cropped_faces = []
        face_boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                # Extract bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                # X1 -->startX_original ,  y1 ---> startY_original ,x2 ---> endX_original , y2 --> endY_original
                (x1, y1, x2, y2) = box.astype("int")

                # Ensure valid coordinates
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                face_boxes.append((x1, y1, x2, y2))

                # Crop and resize the face
                face = image[y1:y2, x1:x2]

                #just to make sure 
                if face.size > 0:
                    face_resized = cv2.resize(face, (299, 299))
                    cropped_faces.append(face_resized)

        return cropped_faces,face_boxes



