import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from src.face_detector import FaceDetector  # Ensure you have a face detection module
import json
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognitionInference:
    def __init__(self, model_path,use_similarity=False, model_type="keras",label_db_path= r"../models/labels.json",embeddings_path=r"..\models\full_embeddings.npz"):
        """
        Initializes the face recognition model for inference.

        Args:
            model_path (str): Path to the trained model file.
            model_type (str): Type of model ("keras", "onnx", or "tflite").
            use_similarity (bool): Whether to use similarity-based recognition (default is False).
            label_db_path (str): Path to the class labels file (JSON).
            embeddings_path (str): Path to the saved embeddings matrix (if using similarity).
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.face_detector = FaceDetector()  # Load face detection module
        self.label_db_path = label_db_path

        self.use_similarity = use_similarity
        self.embeddings_path = embeddings_path

        # self.embeddings_matrix = np.load(embeddings_path)

        # load embedding matrix and embedding labels :
        if use_similarity :
            data_embedding = np.load(self.embeddings_path, allow_pickle=True)
            self.embeddings_matrix = data_embedding["embeddings"]
            self.embeddings_labels = data_embedding["labels"].tolist()  # Convert back to list



        # Load class labels 
        try:
            with open(self.label_db_path, 'r') as f:
                self.class_names = json.load(f)  # List of 105 class names
        except FileNotFoundError:
            raise FileNotFoundError(f"Class names file '{self.label_db_path}' not found.")

    
        # Load the model based on its type
        if self.model_type == "keras":
            self.model = tf.keras.models.load_model(self.model_path)
            if use_similarity :
                self.embedding_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer("dense_7").output)  # Fourth-to-last layer
        elif self.model_type == "onnx":
            # Check if CUDA (GPU) is available for ONNX
            if ort.get_device() == 'GPU':
                providers = ["CUDAExecutionProvider"]
                
            else:
                providers = ["CPUExecutionProvider"]
            self.model = ort.InferenceSession(self.model_path, providers=providers)
        elif self.model_type == "tflite":
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            raise ValueError("Invalid model_type. Choose from 'keras', 'onnx', or 'tflite'.")

    def preprocess_image(self, img):
        """
        Preprocesses an image for inference using InceptionV3 preprocessing.

        Args:
            img (np.array): Cropped face image.

        Returns:
            np.array: Preprocessed image ready for inference.
        """
        # Convert to RGB if necessary (ensure the image has 3 channels)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to the required input size (299x299 for InceptionV3)
        img = cv2.resize(img, (299, 299))
        
        # Convert to float32 and apply InceptionV3 preprocessing
        img = img.astype(np.float32)
        img = preprocess_input(img)  # Normalize the image using InceptionV3's method
        
        # Add batch dimension for model input
        img = np.expand_dims(img, axis=0)
        
        return img

    def detect_and_recognize(self, img_path):
        """
        Detects faces in an image and recognizes them.

        Args:
            img_path (str): Path to the image.

        Returns:
            list: Predicted class IDs for detected faces.
        """

        if isinstance(img_path, str):  # If it's a file path, read it
            img = cv2.imread(img_path)
        else:  # If it's already an image (NumPy array), use it directly
            img = img_path
        

        # Detect faces
        detected_faces ,face_boxes = self.face_detector.detect_faces(img)
        predictions = []

        for face in detected_faces:
            

            # Choose recognition method based on use_similarity flag
            #  and self.embeddings_matrix is not None
            if self.use_similarity:
                embedding = self.compute_embedding(face)
                similarities = self.compute_similarity(embedding)
                
                # Get the most similar face
                best_index = np.argmax(similarities)
                best_score = similarities[best_index]


                # Check threshold and assign label
                if best_score >= 0.5:
                    best_label = self.class_names[str(self.embeddings_labels[best_index])]
                else:
                    best_label = "Non-Defined"


                predictions.append(f"{best_label} ({best_score * 100:.2f}%)")
                
            else :
                processed_face = self.preprocess_image(face)
                pred = self._run_model_inference(processed_face)
                pred = np.argmax(pred)
                predictions.append(self.class_names[str(pred)])

        # Draw bounding boxes & put predicted names
        for idx, (x1, y1, x2, y2) in enumerate(face_boxes):

            # Draw bounding boxes
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add text to the image 
            box_height = y2 - y1
            font_scale = max(0.3, box_height / 420)  # Adjusted for better scaling
            thickness = max(1, int(font_scale * 4))  
            cv2.putText(img, predictions[idx], (x1,(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        return predictions , img


    def _run_model_inference(self, img):
        """Run inference using the model and return the predictions (softmax 105)."""
        # Perform inference based on model type
        if self.model_type == "keras":
            return self.model.predict(img)
        elif self.model_type == "onnx":
            input_name = self.model.get_inputs()[0].name
            return self.model.run(['dense_8'], {input_name: img})[0]
        elif self.model_type == "tflite":
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[1]['index'])
        else:
            raise ValueError("Invalid model type.")
        

    def compute_embedding(self, img):
        """
        Computes the embedding for a given image from the fourth-to-last layer.

        Args:
            img (np.array): Cropped face image.

        Returns:
            np.array: Embedding of the face.
        """
        img = self.preprocess_image(img)

        # Extract the embedding from the fourth-to-last layer (index -4)
        if self.model_type == "keras":
            embedding = self.embedding_model.predict(img)
        elif self.model_type == "onnx":
            input_name = self.model.get_inputs()[0].name
            # Extract output just before the final layer
            embedding = self.model.run(['dense_7'], {input_name: img})[0]
        elif self.model_type == "tflite":
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])  
        else:
            raise ValueError("Invalid model type.")
        
        return embedding

    def compute_similarity(self, embedding):
        """
        Computes the cosine similarity between the computed embedding and the embedding matrix.

        Args:
            embedding (np.array): The embedding to compare.

        Returns:
            np.array: Cosine similarity scores with the embedding matrix.
        """
        # Normalize the embedding
        embedding = embedding.reshape(1, -1)  # Reshape for cosine_similarity

        # Use sklearn's cosine_similarity to compare with the embedding matrix
        similarity_scores = cosine_similarity(embedding, self.embeddings_matrix)

        return similarity_scores.flatten()  # Flatten to make it a 1D array for easier handling
