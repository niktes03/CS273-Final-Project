# Facial Emotion Recognition with FER2013

### CS273P Final Project

A deep learning model that classifies facial expressions into 7 emotion categories using a fine-tuned EfficientNet-B0 pretrained on ImageNet.

**Emotions:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Project Overview

We fine-tune EfficientNet-B0 on the FER2013 dataset to perform 7-class facial emotion recognition. The project includes a systematic ablation study across 4 configurations to analyze the contribution of backbone unfreezing, data augmentation, and class-weighted loss on model performance.

### Key Techniques
- Transfer learning with EfficientNet-B0 pretrained on ImageNet
- Data augmentation (random flips, rotation, color jitter, affine transforms)
- Weighted cross-entropy loss to handle class imbalance (Disgust has ~11x fewer samples than Happy)
- Cosine annealing learning rate scheduler
- Ablation study across 4 experimental configurations


## Dataset

**FER2013** — Facial Expression Recognition 2013

- ~35,000 grayscale 48x48 face images
- 7 emotion classes
- Source: [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

### Download Instructions
1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to your profile → Settings → API → Create New Token
3. Create a file called `kaggle.json` and paste: {"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_API_TOKEN"}. Make sure to replace it with your username and the token you create. 
4. The first cell of `fer2013_emotion_recognition.ipynb` will prompt you to upload `kaggle.json` and download the dataset automatically


## Setup Instructions

This project is designed to run on **Google Colab** with a free T4 GPU.

### Step 1 — Open the notebook in Colab
"CS_273_FINAL_PROJECT.ipynb" from GitHub → "Open in Colab" (top left). 

### Step 2 — Enable GPU
Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save

### Step 3 — Install dependencies
Run the first cell, it installs all required packages automatically:
```
!pip install -q kaggle torchmetrics matplotlib seaborn
```

All other dependencies (torch, torchvision, numpy, pandas, PIL, sklearn) come pre-installed on Colab.


## How to Train the Model

1. Open `fer2013_emotion_recognition.ipynb` in Colab
2. Enable T4 GPU (Runtime → Change runtime type)
3. Run all cells in order from top to bottom
4. When prompted, upload your `kaggle.json` file to download the dataset
5. The notebook will automatically run all 4 ablation experiments

**Expected training time per experiment:** ~10–15 minutes on Colab T4 GPU

The best model weights are saved automatically as `best_model.pth` at the end.


## How to Evaluate the Model

Evaluation runs automatically at the end of each experiment inside the main notebook. The following are generated:

- Overall accuracy on the test set
- Per-class accuracy for all 7 emotions
- Confusion matrix (raw counts + normalized)
- Full classification report (precision, recall, F1)
- Training curves for all 4 ablation experiments


## Ablation Study

We run 4 experiments to measure the contribution of each component:

| # | Configuration | Val Acc | Test Acc |
|---|---|---|---|
| 1 | Baseline (frozen backbone) | 0.4837 | 0.5071 |
| 2 | Unfrozen backbone | 0.6715 | 0.6913 |
| 3 | Unfrozen + Augmentation | 0.7027 | 0.7189 |
| 4 | Unfrozen + Augmentation + Class Weights | 0.7030 | 0.7099 |


## Demo

To run the demo without training the model yourself:

1. Open `fer2013_demo.ipynb` in Colab
2. Enable T4 GPU
3. Run all cells — the trained model downloads automatically
4. Upload any face image when prompted
5. The model outputs the predicted emotion and confidence scores for all 7 classes

Sample images are provided in the `sample_images/` folder for quick testing.


## Expected Outputs

After running the full notebook you should see:
- A class distribution bar chart showing dataset imbalance
- A sample image grid showing 2 examples per emotion
- Training/validation accuracy curves for all 4 experiments
- A confusion matrix on the test set
- Per-class accuracy bar chart
- A saved `best_model.pth` file


## Team

- Rohan Ganesh 
- Sakshi Nikte
- Shikha Patel
