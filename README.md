# ♻️ Automated Waste Classification using Machine Learning  

![Waste Management](https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Waste-separation-bins-Germany.jpg/800px-Waste-separation-bins-Germany.jpg)  

## 📌 Overview  

This project uses **Machine Learning (ML)** and **Computer Vision** to classify waste into different categories, such as **plastic, metal, glass, paper, cardboard, and trash**. 
A **Random Forest Classifier** trained on a dataset of waste images enables automatic classification, supporting sustainable waste management practices.  

---

# Dataset for Training :

https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

# 🔧 Installation

Clone this repository and install the necessary dependencies:
```sh
git clone https://github.com/Glitchtrap991/waste-classification-ml.git  
cd waste-classification-ml
pip install -r requirements.txt
```

If requirements.txt is not available, manually install dependencies:
```sh
pip install numpy opencv-python scikit-learn pickle-mixin matplotlib
```

# 🚀 How It Works

## 1️⃣ Data Preprocessing

Images are resized to 224x224 pixels for uniformity.
Normalization is applied by scaling pixel values between 0 and 1.
Flattened image arrays serve as feature vectors for model training.

```python
import cv2
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (224, 224))
normalized_image = resized_image / 255.0
```
___

## 2️⃣ Feature Reduction using PCA

To handle high-dimensional image data, Principal Component Analysis (PCA) reduces the number of features while retaining essential information.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # Retain 50 principal components
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
```
___

## 3️⃣ Model Training (Random Forest Classifier)

The dataset is used to train a Random Forest Classifier, chosen for its robustness and high accuracy in classification tasks.
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_pca, y_train)
n_estimators=100 → Uses 100 decision trees.
random_state=42 → Ensures reproducibility.
```

The trained model is saved for future use:

```python
import pickle
with open("model_optimized.p", "wb") as f:
    pickle.dump({"model": model, "pca": pca, "categories": categories}, f)
```
___
    
## 4️⃣ Real-Time Image Classification Using Webcam (cv2)

The model is tested using a live webcam feed instead of loading external images:

```python
import cv2
import pickle

# Load trained model and PCA
with open("model_optimized.p", "rb") as f:
    data = pickle.load(f)

model = data["model"]
pca = data["pca"]
categories = data["categories"]

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess captured frame
    img_pca = pca.transform(preprocess_image(frame))

    # Predict category
    prediction = model.predict(img_pca)
    predicted_label = categories[prediction[0]]

    # Display results
    cv2.putText(frame, f"Predicted: {predicted_label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste Classification", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

This script will:
+ ✔ Capture frames from the webcam.
+ ✔ Preprocess them using cv2 and PCA.
+ ✔ Predict the waste category in real-time.
+ ✔ Display the classification label on the screen.
___

## ▶️ How to Run

### Train the Model

```sh
python train_model.py
```

This will preprocess images, train the model, apply PCA, and save the trained model.

### Test the Model

```sh
python test_model.py
```

Press 'q' to exit the webcam feed.
___

## 📈 Results and Performance

Achieved over 65% accuracy using PCA and Random Forest.
Efficient classification with real-time image predictions.
Supports Sustainable Development Goal 12 (UNDP SDG 12) for responsible waste management.
___
