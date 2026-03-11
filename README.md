# CM3070-FINAL-PROJECT-BREAST-CANCER-DETECTION-
CM3070 final project: Breast cancer detection from mammography images using deep learning and Streamlit deployment.


# Overview

This project develops a deep learning system for breast cancer detection from mammography images. The goal is to investigate whether convolutional neural networks (CNNs) can classify mammograms as normal or cancerous.

The project follows an incremental deep learning workflow, beginning with a baseline CNN and progressively applying advanced techniques such as transfer learning, threshold optimisation, fine-tuning, and class imbalance handling.

The final trained model is deployed through a Streamlit web application, allowing users to upload mammography images and obtain predictions in real time.



# Objectives

The objectives of this project are:
	•	Develop a deep learning pipeline for mammography image classification
	•	Evaluate multiple CNN architectures for breast cancer detection
	•	Investigate the impact of class imbalance in medical datasets
	•	Improve cancer detection using threshold optimisation and fine-tuning
	•	Validate the approach using a second mammography dataset
	•	Demonstrate practical usability through a Streamlit deployment



# Datasets

Two mammography datasets were used in this project.

1. DDSM Dataset (Primary Dataset)

The Digital Database for Screening Mammography (DDSM) was used as the primary dataset for model development and optimisation.

Characteristics:
	•	Large mammography screening dataset
	•	Contains normal and cancer cases
	•	Significant class imbalance
	•	Images stored in TFRecord format

This dataset was used to train and evaluate multiple CNN architectures including:
	•	Baseline CNN
	•	Enhanced CNN
	•	Transfer learning models (VGG16, DenseNet121, ResNet50)

Due to the imbalance in screening datasets, evaluation focused on recall and F1-score for the cancer class, rather than accuracy alone.

⸻

2. Breast Mammography Images with Masses Dataset

A second dataset called Breast Mammography Images with Masses was used to validate the pipeline under different dataset conditions.

Characteristics:
	•	Images stored in standard image folders
	•	Binary classification task:
	•	Benign (0)
	•	Malignant (1)
	•	Approximately 10,900 benign images
	•	Approximately 13,700 malignant images

Preprocessing steps included:
	•	Grayscale conversion
	•	CBC image filtering
	•	Image resizing to 224 × 224
	•	Data augmentation applied to the training set

A ResNet50V2 transfer learning model was trained on this dataset.

Final performance:

Accuracy ≈ 92%
ROC-AUC ≈ 0.981




# Model Development Pipeline

The model development process followed a progressive deep learning workflow:
	1.	Baseline CNN
	2.	Enhanced CNN
	3.	Transfer learning architectures
	•	VGG16
	•	DenseNet121
	•	ResNet50
	4.	Threshold optimisation
	5.	Fine-tuning
	6.	Minority class oversampling





# Final Model

The final deployed model is based on Fine-tuned ResNet50 trained on balanced DDSM data.

Key improvements include:
	•	Transfer learning from ImageNet
	•	Fine-tuning of upper network layers
	•	Optimised classification threshold
	•	Balanced training through oversampling



# Results

Final performance on the DDSM dataset:

Accuracy ≈ 95%
Precision (Cancer) ≈ 0.84
Recall (Cancer) ≈ 0.78
F1-Score (Cancer) ≈ 0.81
ROC-AUC ≈ 0.98





# Streamlit Application

The final trained model was deployed using a Streamlit web application.

The application allows users to:
	•	Upload a mammography image
	•	Automatically preprocess the image
	•	Run inference using the trained model
	•	Receive a prediction indicating Normal or Cancer





# Running the Application

Follow these steps to run the Streamlit application.

Clone the repository

git clone https://github.com/ziadallam462/CM3070-FINAL-PROJECT-BREAST-CANCER-DETECTION-.git

Navigate to the project folder

cd CM3070-FINAL-PROJECT-BREAST-CANCER-DETECTION-

Create a virtual environment

python -m venv venv

Activate the environment

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py

Open the application

http://localhost:8501



# Technologies Used
	•	Python
	•	TensorFlow / Keras
	•	Scikit-learn
	•	Streamlit
	•	NumPy
	•	Matplotlib



# Disclaimer

This project is intended for research and educational purposes only.
The system is not a clinical diagnostic tool and should not be used for medical decision-making.


Author

Ziad Allam
University of London
CM3070 Final  Project 
