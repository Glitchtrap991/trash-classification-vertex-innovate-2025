import os
import cv2
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# Set dataset path
dataset_path = "C:\\Users\\Aparajith\\TrashTrain\\archive\\garbage-dataset" #Use your directory where the training images are stored.
categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "biological", "battery", "shoes", "clothes"]

# Function to preprocess a single image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Function to load images in parallel
def load_images(category):
    folder_path = os.path.join(dataset_path, category)
    data, labels = [], []
    if os.path.exists(folder_path):
        for img_file in os.listdir(folder_path):
            try:
                image_path = os.path.join(folder_path, img_file)
                image = cv2.imread(image_path)
                if image is not None:
                    processed_image = preprocess_image(image)
                    data.append(processed_image)
                    labels.append(categories.index(category))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return data, labels

# Load dataset using ThreadPoolExecutor
data, labels = [], []
with ThreadPoolExecutor() as executor:
    results = executor.map(load_images, categories)
    for d, l in results:
        data.extend(d)
        labels.extend(l)

data = np.array(data).astype("float32") / 255.0  # Normalize data
data = data.reshape(len(data), -1)  # Flatten images
labels = np.array(labels)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Reduce to 50 components for efficiency
data = pca.fit_transform(data)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate model
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5)
print(f"Cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%")

# Test accuracy
y_predict = best_model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"Test accuracy: {score * 100:.2f}%")

# Save model
with open("model_optimized.p", "wb") as f:
    pickle.dump({"model": best_model, "categories": categories, "pca": pca}, f)
print("Optimized model saved as 'model_optimized.p'.")

# Real-time webcam prediction
def predict_from_webcam(model, categories, pca):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_image(frame)
        normalized_frame = processed_frame.astype("float32") / 255.0
        flattened_frame = normalized_frame.flatten().reshape(1, -1)
        transformed_frame = pca.transform(flattened_frame)
        prediction = model.predict(transformed_frame)
        label = categories[prediction[0]]
        cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Prediction - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Run real-time prediction
print("Starting webcam prediction...")
predict_from_webcam(best_model, categories, pca)
