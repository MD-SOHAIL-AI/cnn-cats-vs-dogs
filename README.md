# CNN — Cat and Dog Classification

[View the notebook](CNN_Cats_vs_Dogs_Colab.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/MD-SOHAIL-AI/cnn-cats-vs-dogs/blob/main/CNN_Cats_vs_Dogs_Colab.ipynb)

## Goal
Classify cat and dog photos using a neural network with exactly three convolutional layers.

## Dataset
Use the TensorFlow `cats_and_dogs_filtered.zip` download, or attach the `Cat and Dog` dataset by tongpython in Kaggle (using Add Input). The notebook looks for labelled `cats` and `dogs` directories and keeps training, validation and test images separate.

## Method
Read and resize RGB images to 96 × 96, apply random horizontal flips to training samples, train three convolution layers (12/24/48 filters) followed by a small classifier, select best weights using validation loss, and evaluate with accuracy, classification report, confusion matrix and sample predictions. Set for CPU with up to four threads.

## Outputs
The notebook contains plots and printed output. **The pre-filled output is from an offline sanity test on generated coloured patterns, not real pet photos. These numbers must not be reported as Cats vs Dogs accuracy.** Re-run with the actual dataset to obtain authentic results.

## Run
Open the notebook in Google Colab, connect to a CPU runtime and select **Runtime > Run all**. Internet access is needed for the automatic dataset download. In Kaggle, attach the labelled dataset using **Add Input** before running the cells.

For a local Python environment:

```bash
python -m pip install torch numpy matplotlib Pillow scikit-learn jupyter
jupyter notebook CNN_Cats_vs_Dogs_Colab.ipynb
```

Run the cells in order. The dataset is downloaded into `workshop_data/`, and model weights are saved to `outputs/cnn_cats_vs_dogs.pt` locally or `/kaggle/working/cnn_cats_vs_dogs.pt` on Kaggle.

## GitHub
Commit the `.ipynb` and README after a full training run. Do not commit downloaded datasets or large `.pt` files.
