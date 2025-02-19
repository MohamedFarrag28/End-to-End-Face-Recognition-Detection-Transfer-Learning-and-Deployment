# **End-to-End Face Recognition and Detection with Transfer Learning & ONNX Deployment**  

## **Overview**  
This project implements an end-to-end face recognition and detection system using deep learning. It leverages OpenCV for face detection, InceptionV3 for feature extraction, and ONNX for efficient inference. The system is deployed using Streamlit and hosted via Ngrok.  

## **Features**  
✅ Face detection using OpenCV’s DNN module  
✅ Face recognition with InceptionV3 and transfer learning  
✅ ONNX, TFLite, and Keras support for flexible model deployment   
✅ Cosine similarity-based recognition with two modes:
 * Full embedding (more precise but requires more memory)
 * Average embedding (faster, suitable for general cases)
   
✅ Real-time recognition via webcam and image uploads  
✅ Streamlit-based interactive web app  
✅ Hosted using Ngrok for easy remote access  

## **Installation**  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/MohamedFarrag28/End-to-End-Face-Recognition-Detection-Transfer-Learning-ONNX-Deployment.git
   cd End-to-End-Face-Recognition-Detection-Transfer-Learning-ONNX-Deployment
   ```
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**  
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

## **Usage**  
- Upload an image or use a webcam for live face detection and recognition(for now -- > capturing single images).  
- Adjust similarity thresholds for better recognition.  
- View detected faces with predicted names.  

## **Folder Structure**  
```
face_recognition_project/
│── 📂 data/                   # Dataset & preprocessed images
│── 📂 models/                 # Trained models & ONNX files
│── 📂 notebooks/              # Experimentation notebooks
│── 📂 src/                    # Core scripts for inference & detection
│── 📂 streamlit_app/          # Web app interface
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

## **Acknowledgments**  
- OpenCV for face detection  
- TensorFlow/Keras for model training  
- Streamlit for UI development  
- Ngrok for hosting  

🚀 **Ready to recognize faces in real-time!**  

📌 **GitHub Repository:** [End-to-End Face Recognition & Detection](https://github.com/MohamedFarrag28/End-to-End-Face-Recognition-Detection-Transfer-Learning-ONNX-Deployment)
