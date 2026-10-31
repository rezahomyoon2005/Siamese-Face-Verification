# Siamese Network for Face Verification

A PyTorch implementation of a **Siamese neural network** that predicts whether two face images belong to the same person. The project uses a pretrained **EfficientNet-B0** feature extractor and **contrastive loss** to learn a shared embedding space. A small interactive interface built with `ipywidgets` lets users upload and compare two images directly in the Kaggle notebook.

> This is a face **verification** experiment, not a person-identification system. Its predictions are not proof of identity.

## Approach

1. Read identity-labeled images and form positive (same identity) and negative (different identities) pairs.
2. Apply image resizing and light training augmentation (horizontal flips, rotation, and brightness/contrast changes).
3. Process both images with the same EfficientNet-B0 encoder and project them to 256-dimensional embeddings.
4. L2-normalize embeddings and compute their Euclidean distance; train using contrastive loss.
5. Choose a similarity decision threshold using validation pairs, then evaluate on the held-out test split.

A pair is predicted **same person** when its embedding distance is less than or equal to the selected threshold.

## Training configuration

| Setting | Value |
| --- | --- |
| Backbone | EfficientNet-B0 (pretrained) |
| Embedding dimensions | 256 |
| Objective | Contrastive loss (margin 1.0) |
| Optimizer | Adam |
| Initial learning rate | 0.0003 |
| Batch size | 32 |
| Image size | 224 × 224 |
| Epochs | 20 |
| Fine-tuning | Freeze backbone for 5 epochs, then unfreeze last 4 blocks and final convolutional layers |
| Learning-rate schedule | StepLR |
| Decision threshold | Selected by best validation F1 |

## Results

The following metrics are **reported by the saved Kaggle notebook run**, not independently reproduced here.

| Metric | Value |
| --- | ---: |
| Validation F1 | 0.8358 |
| Selected distance threshold | 0.5277 |
| Test accuracy | 0.8088 |
| Test precision | 0.7902 |
| Test recall | 0.8410 |
| Test F1 | 0.8148 |
| Test ROC-AUC | 0.8945 |

The notebook also shows training/validation loss curves, a confusion matrix, and a ROC curve. These metrics are specific to this dataset and its evaluation protocol.

## Dataset

The notebook uses Kaggle's **Face Recognition Dataset - Siamese Network**, mounted at:

```text
/kaggle/input/face-recognition-dataset-siamese-network/dataset/
├── train/
└── val/
```

The provided `train/` directory is split into training and validation **at the image level**; the provided `val/` directory serves as the test set in this notebook. Images from the same identity may therefore appear across training and validation. Results should **not** be interpreted as a guarantee of performance on completely unseen identities.

## Run in Kaggle

1. Import [`Siamese_Face_Verification_GitHub.ipynb`](Siamese_Face_Verification_GitHub.ipynb) into Kaggle.
2. Attach the *Face Recognition Dataset - Siamese Network* dataset.
3. Enable a GPU accelerator. Internet access may be needed initially to install `efficientnet_pytorch` and download pretrained EfficientNet weights.
4. Run cells from top to bottom. The notebook trains the model, selects its validation threshold, evaluates on test pairs, and displays the face comparison widget.
5. Upload two images in the **Face Verification** widget and select **Compare Faces**.

The code uses Python with PyTorch, torchvision, efficientnet-pytorch, scikit-learn, NumPy, Pillow, Matplotlib and ipywidgets.

## Saved model

After training, the notebook writes `best_siamese_model.pth` and `siamese_checkpoint.pth` to `/kaggle/working/`. The latter contains the model state, embedding dimension, image size, and validation-selected threshold. The provided `load_saved_model()` function can reload that checkpoint. The weights are **not included** in this repository unless separately uploaded.

## Limitations and possible improvements

- The model does not automatically detect, crop, or align faces; background and pose changes can affect predictions.
- The image-level train/validation split is not an identity-disjoint evaluation.
- A decision threshold calibrated on validation data may not transfer to other image sources.
- Future work includes face alignment, identity-disjoint testing, and comparing pretrained face-specific encoders.

## License and data

The dataset and pretrained weights may have their own licenses and usage restrictions. Check their terms before redistribution or deployment.
