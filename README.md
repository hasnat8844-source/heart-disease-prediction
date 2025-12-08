# Heart Disease Prediction Using Deep Learning

A production-style **Deep Learning project** that predicts the risk of heart disease using **Artificial Neural Networks (ANN)** with high accuracy. This project demonstrates the real-world potential of AI in healthcare and clinical decision support systems.

---

## 📊 Model Performance

* **Test Accuracy:** 96.09%
* **Validation Accuracy:** 95.73%
* **Training Accuracy:** 96.77%
* **Final Training Loss:** 0.1310
* **Overfitting Status:** Minimal (1.04% gap between training and validation)

These results demonstrate strong model generalization and training stability.

---

## Model Architecture

* **Input Layer:** 13 clinical features
* **Hidden Layers:** 3 Fully Connected Dense Layers
* **Activation Functions:** ReLU (hidden), Sigmoid (output)
* **Regularization:** Dropout (30%) + Batch Normalization
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy

Reference:
TensorFlow Dense Layers Documentation
[https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

---

## Dataset Information

This project uses the **UCI Heart Disease Dataset**, a well-known benchmark dataset in medical machine learning.

Official Source:
UCI Machine Learning Repository
[https://archive.ics.uci.edu/ml/datasets/heart+Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## Development Environment

* **Programming Language:** Python 3.10+
* **Development Tools:** Jupyter Notebook

Reputable References:
Python Official Documentation
[https://www.python.org](https://www.python.org)

Project Jupyter Official Documentation
[https://jupyter.org](https://jupyter.org)

## Training Configuration

* **Epochs:** 60
* **Batch Size:** 32
* **Validation Split:** 20%
* **Early Convergence:** ~25 epochs

Reference:
Google Machine Learning Best Practices
[https://developers.google.com/machine-learning/guides](https://developers.google.com/machine-learning/guides)

---

##  Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Required Libraries

```
tensorflow>=2.10.0  
pandas>=1.5.0  
scikit-learn>=1.2.0  
matplotlib>=3.6.0  
numpy>=1.23.0  
jupyter
```

Reference:
pip Official Documentation
[https://pip.pypa.io/en/stable/](https://pip.pypa.io/en/stable/)

---

## Project Structure

```
heart-disease-prediction/

├── heart_disease_model.ipynb
├── requirements.txt
├── heart.csv
└── README.md
```

---

## Key Features

* StandardScaler for normalization
* Dropout layers to reduce overfitting
* Batch Normalization for faster training
* High generalization performance

Reference:
Scikit-learn StandardScaler Documentation
[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

## Applications

* Clinical decision support research
* Early-stage heart disease risk prediction
* AI in healthcare education

Reputable Reference:
World Health Organization (WHO) – AI in Healthcare
[https://www.who.int/publications/i/item/9789240029200](https://www.who.int/publications/i/item/9789240029200)

---

## Future Enhancements

* Real-time API integration
* Web interface (Flask/FastAPI)
* Model explainability using SHAP

Reference:
SHAP Documentation
[https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)

---

## Developer

**Hasnat**
AI/ML Developer
Specialization: Machine Learning & Deep Learning


