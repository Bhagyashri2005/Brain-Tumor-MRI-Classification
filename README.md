# Brain Tumor MRI Image Classification

## 📌 Project Overview

**Brain Tumor MRI Image Classification** is a deep learning project that classifies brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The project compares a custom Convolutional Neural Network (CNN) with transfer learning models and deploys the final trained model using Streamlit.

---

## 🎯 Objectives

- Preprocess and prepare brain MRI image data.
- Build and train a custom CNN model.
- Apply transfer learning using pretrained deep learning architectures.
- Compare model performance using standard classification metrics.
- Select the final model based on test-set performance.
- Deploy the trained model as an interactive Streamlit application.

---

## 🧠 Classes

| Class | Description |
|---|---|
| Glioma | MRI images belonging to the glioma tumor class |
| Meningioma | MRI images belonging to the meningioma tumor class |
| No Tumor | MRI images without a detected tumor |
| Pituitary | MRI images belonging to the pituitary tumor class |

---

## 📂 Dataset

The dataset contains separate folders for training, validation, and testing.

```text
dataset/
├── train/
├── valid/
└── test/
```

The dataset was obtained from the following Google Drive folder:

**Dataset:** https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju

The dataset is **not included in this GitHub repository** to keep the repository lightweight. Download it separately and place it inside the project directory if you want to reproduce the notebook training workflow.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
Custom CNN
   ↓
Transfer Learning
   ├── MobileNetV2
   ├── Fine-Tuned MobileNetV2
   └── ResNet50
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Model: ResNet50
   ↓
Streamlit Deployment
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit
- Jupyter Notebook
- ResNet50
- Transfer Learning

---

## 🧪 Image Preprocessing

For model training and prediction:

- Images are resized to **224 × 224 pixels**.
- Images are converted to RGB format.
- Data augmentation is applied to the training images.
- ResNet50 preprocessing is used for the final model.

Training augmentation includes:

- Rotation
- Width shift
- Height shift
- Zoom
- Horizontal flip

---

## 🤖 Models Implemented

### 1. Custom CNN

A CNN was developed from scratch using:

- Conv2D layers
- Batch Normalization
- Max Pooling
- Global Average Pooling
- Dense layers
- Dropout
- Softmax output layer

### 2. MobileNetV2

MobileNetV2 pretrained on ImageNet was used for transfer learning with a custom classification head.

### 3. Fine-Tuned MobileNetV2

The pretrained MobileNetV2 model was further fine-tuned by unfreezing selected deeper layers.

### 4. ResNet50

ResNet50 pretrained on ImageNet was used as the final transfer learning model.

The ResNet50 base was initially frozen and a custom classification head was trained for the four MRI classes.

---

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Custom CNN | 32.52% | 34.62% | 32.52% | 24.26% |
| MobileNetV2 | 84.15% | 84.69% | 84.15% | 83.57% |
| Fine-Tuned MobileNetV2 | 77.24% | 80.55% | 77.24% | 75.20% |
| ResNet50 | **88.21%** | **88.18%** | **88.21%** | **87.98%** |

The final deployed model is **ResNet50**.

---

## 🏆 Final Model

**Architecture:** ResNet50  
**Approach:** Transfer Learning  
**Input Size:** 224 × 224 × 3  
**Number of Classes:** 4

### Test Performance

- Accuracy: **88.21%**
- Precision: **88.18%**
- Recall: **88.21%**
- F1-Score: **87.98%**

The reported metrics are based on the project's test dataset and should be interpreted within the limitations of that dataset.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

The application allows the user to:

1. Upload a brain MRI image.
2. Preview the uploaded image.
3. View image information.
4. Analyze the MRI using the trained ResNet50 model.
5. View the predicted class.
6. View prediction confidence.
7. View probability distribution across all four classes.

---

## 📁 Project Structure

```text
Brain_Tumor_MRI/
│
├── app.py
├── best_resnet50.keras
├── requirements.txt
├── Brain_Tumor_MRI.ipynb
├── .gitignore
└── README.md
```

The dataset and intermediate model files are excluded from the GitHub repository.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser, normally at:

```text
http://localhost:8501
```

---

## 📓 Notebook

`Brain_Tumor_MRI.ipynb` contains the complete project workflow, including:

- Dataset exploration
- Image preprocessing
- Data augmentation
- Custom CNN development
- Transfer learning
- Model training
- Model evaluation
- Confusion matrix
- Classification metrics
- Model comparison
- Final model prediction

---

## 📦 Requirements

Main dependencies include:

```text
streamlit
tensorflow==2.20.0
numpy
pillow
```

---

## ⚠️ Disclaimer

This application is intended only for **educational and demonstration purposes**.

It is not a medical device and should not be used to diagnose, treat, or make clinical decisions about brain tumors. Any medical interpretation should be performed by qualified healthcare professionals using appropriate clinical evaluation and diagnostic procedures.

---

## 👩‍💻 Author

**Bhagyashri Patil**

Computer Engineering  
Interested in Software Development, Backend Development, AI/ML, and Deep Learning.
