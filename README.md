# 🖋️ MNIST Digit Recognition

## 📌 Project Overview
This project implements **handwritten digit recognition** using the **MNIST dataset**.  
It demonstrates both **traditional Machine Learning** and **Deep Learning** approaches:  
- **Random Forest Classifier** (Traditional ML)  
- **Convolutional Neural Network (CNN)** (Deep Learning)  

The project also includes **data augmentation** to improve model generalization and testing with **custom handwritten digits**.

---

## 🎯 Objectives
- Understand the difference between ML and Deep Learning approaches for image classification.
- Learn image preprocessing techniques for model readiness.
- Explore **Random Forest** as a classical ML algorithm.
- Implement a **CNN** for improved accuracy and robustness.
- Perform **Data Augmentation** to reduce overfitting.
- Test the model on **real-world custom digits**.

---

## 📂 Dataset
**MNIST Dataset** — 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

Source: [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)  
Available via:  
```python
from tensorflow.keras.datasets import mnist
```

---

## 🛠️ Technologies Used
- **Python**
- **TensorFlow / Keras**
- **scikit-learn**
- **NumPy**
- **Matplotlib**
- **Pandas**

---

## ⚙️ Workflow
1. **Load Dataset**  
2. **Preprocess Images** (Normalization, Reshaping)  
3. **Data Augmentation** (Rotation, Zoom, Shift)  
4. **Train Random Forest Classifier**  
5. **Train CNN Model**  
6. **Evaluate Models on Test Data**  
7. **Predict Custom Digit Image**  

---

## 📊 Model Performance
| Model              | Test Accuracy |
|--------------------|--------------|
| Random Forest      | ~96%         |
| CNN (Augmented)    | ~99%         |

---

## 🖼️ Custom Digit Prediction
- A test was conducted using a custom **digit.png** image.
- **CNN** correctly predicted the digit.  
- **Random Forest** misclassified the digit as '9'.  

**Conclusion:** CNN is better suited for image-based pattern recognition due to its ability to capture spatial features.

---

## 🚀 How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook or script:
```bash
jupyter notebook MNIST_Digit_Recognition.ipynb
```

---

## 📈 Learning Outcomes
- Gained practical experience in both **ML** and **Deep Learning** approaches.
- Understood **data preprocessing** for image datasets.
- Learned **Data Augmentation** techniques to improve generalization.
- Observed **real-world testing** challenges with custom inputs.

👩‍💻Author  : 
  SHALINI SAURAV
