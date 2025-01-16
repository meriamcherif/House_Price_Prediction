# House Price Prediction Using Neural Networks

This project serves as a practice ground to improve PyTorch skills while solving a classic regression problem: predicting house prices. The focus is on implementing foundational machine learning concepts such as embedding layers for categorical features, normalization for continuous features, and creating a FeedForward Neural Network (FFNN).


## Overview

The goal of this project is not just to predict house prices but to deepen understanding of PyTorch by:
- Working with embeddings for categorical variables.
- Using batch normalization and dropout for better model generalization.
- Building and training a FeedForward Neural Network from scratch.
- Handling both categorical and continuous data.

The dataset includes features such as:
- **Categorical features**: MSSubClass, MSZoning, Street, LotShape.
- **Continuous features**: LotFrontage, LotArea, YearBuilt, 1stFlrSF, 2ndFlrSF.
- **Target variable**: SalePrice.

---

## Technologies Used

- **Python** for coding and data preprocessing.
- **PyTorch** for building and training the neural network.
- **Pandas & NumPy** for data manipulation.
- **Matplotlib** for visualizing results.

---

## Project Workflow

1. **Data Preprocessing**:
   - Handled missing values.
   - Encoded categorical features using `LabelEncoder`.
   - Normalized continuous features using `BatchNorm1d`.

2. **Feature Engineering**:
   - Embedded categorical features into dense representations using `nn.Embedding`.

3. **Model Architecture**:
   - Created a custom FeedForward Neural Network with PyTorch.
   - Applied dropout and batch normalization for regularization and stable training.

4. **Training**:
   - Used Mean Squared Error (MSE) as the loss function.
   - Optimized the model using Adam optimizer.
   - Evaluated performance with Root Mean Squared Error (RMSE).

5. **Testing**
6. **Saving the model and its weights**