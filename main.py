import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. LOAD DATASET
digits = datasets.load_digits()
X = digits.images   # images (8x8 grayscale)
y = digits.target   # labels 0–9

print("Dataset loaded. Total images:", len(X))

# 2. PREPROCESSING FUNCTION USING OPENCV
def preprocess_image(img):
    # resize to 32x32
    img_resized = cv2.resize(img, (32, 32))

    # normalize (0–1)
    img_norm = img_resized / 16.0

    # convert to uint8 for OpenCV image ops
    img_uint8 = np.uint8(img_norm * 255)

    # brightness & contrast
    alpha = 1.5  # contrast
    beta = 20    # brightness
    enhanced = cv2.convertScaleAbs(img_uint8, alpha=alpha, beta=beta)

    # Gaussian blur filtering
    filtered = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return filtered

# apply preprocessing to all images
X_processed = np.array([preprocess_image(img) for img in X])

# flatten for ML model
X_flattened = X_processed.reshape(len(X_processed), -1)

# 3. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_flattened, y, test_size=0.2, random_state=42
)

# 4. TRAIN MODEL (SVM)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 5. PREDICT
y_pred = model.predict(X_test)

# 6. EVALUATE
acc = accuracy_score(y_test, y_pred)
print("Model accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# 7. SHOW ORIGINAL vs ENHANCED IMAGES
fig, axes = plt.subplots(1, 2)
axes[0].imshow(X[0], cmap='gray')
axes[0].set_title("Original")
axes[1].imshow(preprocess_image(X[0]), cmap='gray')
axes[1].set_title("Enhanced")
plt.show()

# 8. SHOW SAMPLE PREDICTIONS
for i in range(5):
    plt.imshow(X_test[i].reshape(32, 32), cmap='gray')
    plt.title(f"Actual: {y_test[i]}  Predicted: {y_pred[i]}")
    plt.axis('off')
    plt.show()
