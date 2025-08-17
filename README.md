# Vision Transformer Notebook

This folder contains the `visionTransformer.ipynb` notebook, which demonstrates a Vision Transformer (ViT) implementation from scratch for image classification using the MNIST dataset.

## Contents
- `visionTransformer.ipynb`: Jupyter notebook with code, explanations, and results for training and evaluating a Vision Transformer model.

## Dataset
- **MNIST**: Handwritten digit dataset (28x28 grayscale images, 10 classes).
- Data files are located in `../data/raw/MNIST/raw/`.

## Model
- **Vision Transformer (ViT)**: Implements the transformer architecture for image classification.
- Key components:
  - Patch embedding
  - Positional encoding
  - Transformer encoder blocks
  - Classification head

## Usage
1. Open `visionTransformer.ipynb` in Jupyter Notebook or VS Code.
2. Ensure MNIST data is available in the specified path.
3. Run the notebook cells sequentially to:
   - Load and preprocess data
   - Build the ViT model
   - Train and evaluate the model

## Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

You can install dependencies using:
```bash
pip install torch torchvision numpy matplotlib
```

## Results
- The notebook reports training and test accuracy, and visualizes predictions.
- You can modify hyperparameters and model architecture for experimentation.

## References
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Author
- [chitrakumarsai](https://github.com/chitrakumarsai)

---
For questions or issues, please open an issue in the main repository.
