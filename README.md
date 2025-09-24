# Neura 
# Brain tumor detection model
          
          
          
          
                            <img width="597" height="570" alt="image" src="https://github.com/user-attachments/assets/94ea2a49-06b5-49b9-b7c3-32398a128df0" />




This project builds a deep learning model to classify brain tumor MRI scans into different categories using transfer learning (Xception model). The goal is to assist doctors and radiologists by providing an automated, accurate, and fast diagnostic tool.

📌 Problem Statement

Brain tumors are a serious health issue, and timely detection is crucial. Manual diagnosis of MRI scans is time-consuming and prone to human error. This project utilizes Convolutional Neural Networks (CNNs) to classify MRI images of brain tumors automatically.

📂 Dataset

MRI images are organized into folders:

Training/

Testing/

Each folder contains subfolders for different tumor classes:

Glioma

Meningioma

Pituitary Tumor

No Tumor

⚙️ Approach
1. Data Preprocessing

Resized all images to 224 × 224 pixels.

Normalized pixel values.

Split into train / validation / test sets.

Used ImageDataGenerator for batching and preprocessing.

2. Model Architecture

Backbone: Xception (pretrained on ImageNet).

Custom top layers:

Dropout (0.5)

Dense (128, ReLU)

Dropout (0.25)

Dense (Softmax for multi-class output)

Optimizer: Adamax (lr = 0.001)

Loss: Categorical Crossentropy

Metrics: Accuracy

3. Training

Trained for 10 epochs.

Monitored training vs validation accuracy & loss.

4. Evaluation

Metrics: Accuracy, Confusion Matrix, ROC-AUC, Classification Report.

Final model saved as brain_tumor_model.h5 for deployment.

📊 Results

Achieved strong performance with transfer learning.

Training and validation accuracy curves showed good convergence.

Model is ready for deployment in real-world applications.

🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook brain_tumor_model.ipynb


After training, the model will be saved as:

brain_tumor_model.h5

📈 Future Improvements

Add data augmentation to improve generalization.

Experiment with other architectures like EfficientNet.

Use Grad-CAM for model explainability.

Deploy as a Flask / FastAPI web app for real-world usage.

🛠️ Tech Stack

Python

TensorFlow / Keras

Scikit-learn

Pandas / NumPy

Matplotlib / Seaborn

🙌 Acknowledgments

Pretrained Xception model from TensorFlow.

Brain MRI dataset (publicly available on Kaggle and similar sources).
