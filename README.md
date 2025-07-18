# agricultural-classification

🌾 Agricultural Crop Classifier

This project is a deep learning-based image classification system built to identify different types of agricultural crops using computer vision. It uses a fine-tuned ConvNeXt-Tiny model trained on categorized crop images and deployed via a Streamlit web application.

📁 Project Structure

.
├── dataset/
│   └── train/
│       ├── almond/
│       ├── banana/
│       ├── cardamom/
│       ├── Cherry/
│       └── ... (30 total crop folders)
├── Agriculture.py      # Model training script (PyTorch)
├── app.py              # Streamlit app for prediction
├── convnext_agriculture_model.pt  # Saved model weights
└── README.md           # Project documentation

🌱 Supported Crop Categories (30)

- almond
- banana
- cardamom
- Cherry
- chilli
- clove
- coconut
- Coffee-plant
- cotton
- Cucumber
- Fox_nut (Makhana)
- gram
- jowar
- jute
- Lemon
- maize
- mustard-oil
- Olive-tree
- papaya
- Pearl_millet (bajra)
- pineapple
- rice
- soyabean
- sugarcane
- sunflower
- tea
- Tobacco-plant
- tomato
- vigna-radiati (Mung)
- wheat

🏗️ How It Works

1. Training the Model
- Run Agriculture.py to train a ConvNeXt-Tiny model on your dataset.
- It uses data augmentation, validation splitting, early stopping, and saves the best model to convnext_agriculture_model.pt.

2. Running the App
- Launch the Streamlit app using:
  streamlit run app.py
- Upload an image of a crop to get its predicted class and model confidence.

🧠 Model Details

- Architecture: ConvNeXt-Tiny (pretrained on ImageNet)
- Framework: PyTorch
- Augmentations: Resize, Rotation, Horizontal Flip, Color Jitter
- Loss Function: CrossEntropyLoss with label smoothing
- Optimizer: AdamW
- Early Stopping: Triggered after 5 epochs without improvement

🔧 Requirements

Install dependencies using:
pip install -r requirements.txt

Required packages include:
- torch
- torchvision
- streamlit
- Pillow

📈 Sample Output

After uploading an image, the app returns:
- ✅ Predicted Class: e.g., soyabean
- 📈 Confidence: e.g., 98.34%

📸 Dataset Notes

Ensure the dataset is structured as:
dataset/train/
    ├── class_1/
    ├── class_2/
    └── ...

Each subfolder should contain relevant crop images in .jpg, .png, etc.

📬 Contributions

Feel free to fork, modify, or raise issues for improvements. Pull requests are welcome!

Auto-uploaded project.
