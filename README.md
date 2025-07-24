# 🌾 ConvNeXt Agricultural Crop Classifier

A deep learning-based image classification tool to identify different types of agricultural crops using the ConvNeXt-Tiny architecture, trained with PyTorch and deployed using Streamlit.

---

## 📂 Project Structure

```
.
├── dataset/
│   └── train/               # Subfolders for each crop category (30 total)
├── Agriculture.py           # Model training script (PyTorch)
├── app.py                   # Streamlit app for prediction
├── convnext_agriculture_model.pt  # Trained model weights (saved after best validation)
└── README.txt               # Project documentation (this file)
```

---

## 🌱 Crop Categories Supported

- almond, banana, cardamom, Cherry, chilli, clove, coconut, Coffee-plant, cotton, Cucumber
- Fox_nut (Makhana), gram, jowar, jute, Lemon, maize, mustard-oil, Olive-tree, papaya
- Pearl_millet (bajra), pineapple, rice, soyabean, sugarcane, sunflower, tea, Tobacco-plant
- tomato, vigna-radiati (Mung), wheat

---

## ⚙️ Training Pipeline

- **Script:** `Agriculture.py`
- **Architecture:** ConvNeXt-Tiny (ImageNet pretrained)
- **Augmentations:** Resize, Rotation, Flip, Color Jitter
- **Validation Split:** 20%
- **Loss Function:** CrossEntropyLoss (label smoothing 0.1)
- **Optimizer:** AdamW (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau
- **Early Stopping:** After 5 epochs without improvement

Trained model is saved as `convnext_agriculture_model.pt` after best validation accuracy.

---

## 🖼️ Web App (Streamlit)

- **Script:** `app.py`
- **Framework:** Streamlit
- **Function:** Upload crop image and get prediction + confidence

### Run it with:
```bash
streamlit run app.py
```

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Required packages include:

- torch
- torchvision
- streamlit
- Pillow

---

## 📈 Sample Prediction

After uploading an image:

- ✅ Predicted Class: e.g., *soyabean*
- 📉 Confidence Score: e.g., *97.82%*

---

## 📸 Dataset Format

Your dataset should look like:

```
dataset/train/
├── almond/
├── banana/
├── ...
```

Each folder contains images of the respective crop class.

---

## 🤝 Contribution

Open to improvements, bug fixes, and feature suggestions. Feel free to fork and submit PRs.

---

---

## 📖 Author

* **C B Vinay**
* GitHub: [cbvinay-7](https://github.com/cbvinay-7)

---

## 🎉 Acknowledgements

- PyTorch team
- Streamlit team for fast app prototyping
- ConvNeXt paper authors

---

## 🛡️ License

This project is licensed under the MIT License.

---

## 📬 Contact

For support or collaboration, reach out via GitHub or your preferred channel.

Uploaded via Streamlit GitHub Uploader.
