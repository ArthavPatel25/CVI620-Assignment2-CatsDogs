CVI620 – Assignment 2
Cat vs Dog Image Classification (Simple CNN + VGG16 Transfer Learning)

This repository contains my implementation for Question 1 of CVI620 Assignment 2, where the goal was to classify images of cats and dogs using multiple machine learning methods, compare their performance, and choose the best model.

- Project Overview

Two models were trained on the provided Cat/Dog dataset:

1. Simple CNN (from scratch)

3 convolution + max-pool layers

Works on normalized images

Achieved ~75% validation accuracy

2. VGG16 Transfer Learning (ImageNet)

Pretrained VGG16 convolution base (frozen)

Custom classifier head: Flatten → Dense(256) → Dropout → Softmax

Correct ImageNet preprocessing applied

Achieved ~97% validation accuracy

VGG16 clearly outperformed the scratch CNN, so it was saved as the final classifier.

- Repository Structure
CVI620-Assignment2-CatsDogs/
│
├── CatsDogs_Assignment2.ipynb      # Full training + testing notebook
├── model/
│     └── cats_dogs_best.keras      # Final saved model (VGG16)
└── test_images/                    # Internet images used for inference (optional)

- How to Run Training (Optional)

If running locally or in Colab:

python train.py


Or simply open the notebook:

CatsDogs_Assignment2.ipynb


All training code for both the Simple CNN and VGG16 is included.

- Running Inference

Use the inference.py script or run the inference cell inside the notebook.

python inference.py path_to_image.jpg


Example output:

Prediction: Dog (Confidence: 0.982)
