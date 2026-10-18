# Deep Learning – Lab 1: Classification and Regression with PyTorch (MLP/DNN)

**Course:** Deep Learning – Pr. Elaachak Lotfi  
**Institution:** Faculté des Sciences et Techniques de Tanger, Université Abdelmalek Essaadi  
**Author:** Lorraine Agnès  
**Tools:** Kaggle Notebooks, PyTorch, scikit-learn, skorch, pandas, matplotlib, seaborn, GitHub

## Objective

Get familiar with the PyTorch library by building DNN/MLP architectures for a regression task (Part 1) and a multi-class classification task (Part 2).

## Repository content

| File | Description |
|---|---|
| `part1_regression.ipynb` | Part 1: regression on the NYSE dataset |
| `part2_classification.ipynb` | Part 2: multi-class classification (to be completed) |
| `images/` | Figures used in this report |

## Part 1 – Regression (NYSE stock prices)

**Dataset:** [New York Stock Exchange](https://www.kaggle.com/datasets/dgawlik/nyse), file `prices-split-adjusted.csv` (about 850,000 daily records for about 500 companies; columns: date, symbol, open, close, low, high, volume).

**Task:** predict the closing price `close` from `open`, `low`, `high` and `volume`.

### 1. Exploratory Data Analysis

- No missing values.
- `open`, `low`, `high` and `close` are almost perfectly correlated (correlation above 0.99), so the regression problem is expected to be easy.
- `volume` has a very skewed distribution and a scale very different from the prices.
- Consequences for preprocessing: a `log1p` transform on `volume` and standardization of all variables.

### 2. Baseline MLP

- **Preprocessing:** `log1p(volume)`, random 80/20 train/test split (`random_state=42`), `StandardScaler` fitted on the training set only (to avoid data leakage) and applied to features and target.
- **Architecture:** 4 → 64 → 32 → 1, ReLU activations, no activation on the output layer.
- **Training:** MSE loss, Adam optimizer, learning rate 1e-3, batch size 1024, 20 epochs.
- **Metric:** since "accuracy" is not meaningful for a regression, the **R² score** is used in place of the accuracy curve requested in the lab.
- **Result:** R² ≈ 0.9999 on both train and test sets.

### 3. Hyperparameter search (GridSearch)

`GridSearchCV` from scikit-learn, with PyTorch models wrapped by `skorch` (`NeuralNetRegressor`). 3-fold cross-validation, R² scoring, on a random subsample of 50,000 training rows to keep the search fast (32 combinations × 3 folds = 96 fits).

| Hyperparameter | Values tested |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimizer | Adam, SGD |
| Epochs | 10, 20 |
| Neurons per hidden layer | 32, 64 |
| Hidden layers | 1, 2 |

**Best configuration:** Adam, learning rate 0.01, 20 epochs, 2 hidden layers of 64 neurons (cross-validated R² = 0.99986).

### 4. Learning curves (Loss/Epochs and R²/Epochs)

Baseline model:

![Baseline curves](images/curves_base.png)

GridSearch model, retrained on the full training set:

![GridSearch curves](images/curves_best.png)

**Interpretation**

- **No overfitting:** train and test curves are almost identical for both models.
- **Baseline model:** the loss reaches its low level in the very first epochs, then fluctuates (R² between 0.99973 and 0.99993). These oscillations come from mini-batch noise with a constant learning rate; they are tiny in absolute terms (about 1e-4 on R²).
- **GridSearch model:** clearly less stable. The loss shows spikes (up to about 6e-3) and the R² drops as low as 0.994 (epoch 17), and ends around 0.9992, below the baseline.
- **The "best" configuration is therefore not better in practice.** Two likely reasons: (1) the search ran on a 50,000-row subsample, where one epoch performs far fewer weight updates than on the full training set, so a learning rate of 0.01 that looked acceptable becomes too aggressive on the full data; (2) the cross-validation score is measured after the last epoch, which is noisy because of the oscillations, and the differences between configurations (5th decimal of R²) are of the same order as this noise.

### 5. Regularization techniques and comparison

Seven variants of the same MLP (4 → 64 → 32 → 1, Adam, lr = 1e-3, 20 epochs) were trained under identical conditions. A validation set (10% of the training set) is used by the scheduler and early stopping, so the test set is never used for decisions.

| Model | R² train | R² test | MSE test | Epochs run | Best epoch (val) | Instability* | Time (s) |
|---|---|---|---|---|---|---|---|
| 1. Baseline | 0.999926 | 0.999924 | 0.000077 | 20 | 17 | 0.000013 | 122.7 |
| 2. Weight decay (1e-4) | 0.999813 | 0.999809 | 0.000193 | 20 | 16 | 0.000035 | 123.3 |
| 3. Dropout (0.1) | 0.998209 | 0.998212 | 0.001801 | 20 | 1 | 0.000443 | 123.4 |
| 4. BatchNorm | 0.999538 | 0.999533 | 0.000471 | 20 | 13 | 0.000179 | 126.4 |
| 5. LR scheduler (ReduceLROnPlateau) | 0.999927 | 0.999925 | 0.000076 | 20 | 16 | 0.000005 | 121.8 |
| 6. Early stopping (patience 5) | 0.999935 | 0.999934 | 0.000067 | 20 | 17 | 0.000013 | 121.5 |
| 7. Combined (BatchNorm + weight decay + scheduler + early stopping) | 0.999751 | 0.999750 | 0.000252 | 10 | 5 | 0.000297 | 65.7 |

\*Standard deviation of the test R² over the last 5 epochs (lower is more stable).

![Regularization comparison](images/regularisation_comparaison.png)

**Analysis**

- **No overfitting to fix:** the train/test R² gap is at most 5e-6 for every variant, so regularization had no generalization gap to reduce on this task.
- **Best results:** early stopping (lowest test MSE, 0.000067), the learning-rate scheduler (the most stable, instability 5e-6) and the baseline are practically equivalent (R² test ≈ 0.99992). The differences (about 1e-5) come from a single run and should not be considered significant.
- **Early stopping:** with patience 5 it never triggered in the standalone run (20 epochs executed); its small gain comes from restoring the weights of the best validation epoch (17) instead of the last epoch.
- **Scheduler:** reducing the learning rate when the validation loss stagnates gives the smoothest curves.
- **Dropout, BatchNorm and weight decay degrade the results.** Likely explanations (not tested): the network is small and the target is an almost deterministic function of the inputs, so Dropout adds noise to the output and BatchNorm adds batch-dependent noise, while weight decay pulls the weights away from the values needed for an almost linear mapping.
- **Combined model:** worse than the baseline. Early stopping triggered at epoch 10 (best epoch 5), probably because the validation loss of the BatchNorm network is noisy. It is the fastest model (65.7 s), but at the cost of accuracy.

### 6. Conclusions and limitations

- A small MLP predicts `close` with R² ≈ 0.9999. The problem is easy because `close` is strongly determined by `open`, `low` and `high`.
- On this problem, regularization improves **stability** (scheduler) and model selection (early stopping with best-weights restoration), not generalization.
- Hyperparameters selected on a subsample did not transfer to full-data training.
- Limitations: random split instead of a chronological split (a real forecasting setting would predict future prices), single random seed (no confidence intervals), a search grid that is small and whose best point lies on its boundary, and R² used as a substitute for accuracy.

## Part 2 – Multi-class classification (Predictive Maintenance)

*To be completed.*

## What I learned

- Building and training an MLP in PyTorch: tensors, `Dataset`/`DataLoader`, the training loop and the evaluation loop.
- Fitting the scaler on the training data only to avoid data leakage, and keeping a separate validation set so that the test set is never used for decisions.
- Running a hyperparameter search with scikit-learn and skorch, and its pitfalls: a search on a subsample may not transfer to full-data training, and noisy scores make small differences meaningless.
- Reading learning curves (overfitting, convergence, oscillations) and choosing the right scale for the axes.
- Regularization is not an automatic improvement: its usefulness depends on whether the model actually overfits.

## How to run

1. Open the notebooks on Kaggle and add the datasets as inputs: [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse) and [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).
2. Enable Internet in the notebook settings (needed to install `skorch`).
3. Run all cells from top to bottom.
