# 🩻 COVID-19 & Pneumonia Detection from Chest X-ray Images

This project implements a Convolutional Neural Network (CNN) using **TensorFlow Keras** to detect Pneumonia (including COVID-19-related cases) from chest X-ray images. The model is trained on a publicly available X-ray dataset and achieves strong performance for binary classification.

This work is part of an AI/ML Developer Technical Test.

---

## 🚀 Project Overview

- **Goal:** Classify chest X-ray images into:
  - PNEUMONIA
  - NORMAL

- **Frameworks/Libraries:**
  - TensorFlow
  - Keras
  - OpenCV
  - Pillow
  - Matplotlib
  - Seaborn
  - Scikit-learn

- **Input Images:** Grayscale, resized to 224×224 pixels

---

## 📂 Dataset

- Dataset used:
  - [`Chest X-Ray Images (Pneumonia)` on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Folder structure:
    ```
    xray_dataset_covid19/
        train/
            PNEUMONIA/
            NORMAL/
        test/
            PNEUMONIA/
            NORMAL/
    ```

---

## 🏗️ Model Architecture

The CNN architecture: 

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling → Dropout
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling → Dropout
- Flatten
- Dense (128) → ReLU
- Dense (1) → Sigmoid

REFERENCES: Kaggle competetion notebooks to build the most relialble model for our task, and epoch count 15 was fixed by going throug multiple Kaggle notebooks.

**Compilation:**

- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metrics: Accuracy  

---

## ⚙️ Preprocessing & Augmentation

- Color Mode: Grayscale  
- Image size: 224×224  
- Pixel normalization: `1./255`  
- Data augmentation via:
    ```python
    ImageDataGenerator(rescale=1./255)
    ```

---

## 🧪 Training

- Epochs: 15  
- Batch size: 64  
- Early stopping defined but not used in final training

Example:
```python
history = model.fit(
    train_generation,
    epochs=15,
    batch_size=64,
    validation_data=test_generation
)
```

---

## 📈 Performance

- **Test Accuracy:** ~98%  
- Confusion matrix computed on test set using `confusion_matrix()` from scikit-learn

Example:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_label, model_pred)
print(cm)
```

---

## 📸 Sample Image Prediction

Three test images were loaded manually for single prediction outside of the generator pipeline.

Sample workflow:
```python
image = Image.open(image_path).convert("RGB")
image_np = tf.keras.utils.img_to_array(image)
gray = cv.cvtColor(image_np, cv.COLOR_RGB2GRAY)
resized = cv.resize(gray, (224, 224))
normalized = resized / 255.0
img_input = normalized.reshape(1, 224, 224, 1)
```

Prediction and classification:
```python
predict = model.predict(img_input)[0][0]
if predict >= 0.5:
    print("PNEUMONIA")
else:
    print("NORMAL")
```

Sample output:
```
Case detected is "PNEUMONIA"
With accuracy of 99.34%

Case detected is "NORMAL"
With accuracy of 97.56%
```

---

## 💾 Saving the Model

The trained model is saved as:
```
COVID-19 Xray Detection.h5
```

Reload it using:
```python
model = tf.keras.models.load_model('COVID-19 Xray Detection.h5')
```

---

## 🧪 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python image_detection.py
```

Make sure your dataset folder is named `xray_dataset_covid19` and correctly placed relative to your script.

---

## 🧰 Example `requirements.txt`

I would recommend installing the entire requirements for these 3 attached projects rather than installing them individually. 
But if you need to install requirements indicidually, use this to create another requirements.txt file by copying these dependencies.

```text
numpy
pandas
matplotlib
seaborn
opencv-python
Pillow
tensorflow
scikit-learn
```

---

## 📝 Notes

- You must manually change image paths if running prediction locally.
- The model can be integrated into a Streamlit app or deployed with Docker.
- Currently, early stopping and callbacks are optional but supported.

---

## ✅ Summary

This project demonstrates a clean and effective CNN-based pipeline to detect pneumonia from X-ray scans, achieving high accuracy and minimal false classifications. It can serve as the backend for a deployable medical assistant or AI diagnostics tool.

---

## 🧠 Author

**Anosh Andrews**  
AI/ML Developer  