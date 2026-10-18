# Deep Learning Lab 1: PyTorch DNNs for NYSE Stock Regression and Predictive Maintenance Classification

**Student:** RAHELIARISOA Andriamasy Lorraine Agnès  
**Program:** Engineering Cycle, Software and Intelligent Systems, Computer Science Department, Faculté des Sciences et Techniques de Tanger, Université Abdelmalek Essaadi  
**Course:** Deep Learning and Applications  
**Supervisor:** Pr. Elaachak Lotfi  
**Academic year:** 2026-2027  
**Tools:** Kaggle Notebooks, PyTorch, scikit-learn, skorch, imbalanced-learn, pandas, matplotlib, seaborn, GitHub

## Objective

In this lab I learn to build MLP (DNN) models with PyTorch. Part 1 is a regression task on stock prices. Part 2 is a multi-class classification task on machine failures.

## Repository content

| File | Description |
|---|---|
| `part1_regression.ipynb` | Part 1: regression on the NYSE dataset |
| `part2_classification.ipynb` | Part 2: multi-class classification on the Predictive Maintenance dataset |
| `images/` | Figures used in this report |

## Part 1: Regression (NYSE stock prices)

**Dataset:** [New York Stock Exchange](https://www.kaggle.com/datasets/dgawlik/nyse). I used the file `prices-split-adjusted.csv`: about 850,000 daily rows for 501 companies (date, symbol, open, close, low, high, volume).

**Goal:** predict the closing price `close` from `open`, `low`, `high` and `volume`.

### 1. Exploratory data analysis

- There are no missing values.
- Prices are very skewed to the right. Most values are below 200, but a few reach about 1,600.
- Volume is even more skewed. A few days have hundreds of millions of shares.
- `open`, `low`, `high` and `close` are almost the same thing (correlation above 0.99), so I expect the regression to be easy.
- The variables have very different scales, so I need to standardize them. I also apply `log1p` to volume.

### 2. Baseline MLP

- **Data:** `log1p` on volume, random 80/20 split (`random_state=42`), and a `StandardScaler` fitted on the training set only (to avoid data leakage) and applied to the features and the target. A random split is simple and fine for this lab. For real forecasting I would split by date.
- **Model:** 4 → 64 → 32 → 1, ReLU activations, no activation on the output.
- **Training:** MSE loss, Adam, learning rate 1e-3, batch size 1024, 20 epochs.
- **Metric:** accuracy does not make sense for a regression, so I use the **R² score** instead of the accuracy curve asked in the lab.
- **Result:** R² is about 0.9999 on both train and test, and the two losses are almost equal, so there is no overfitting. The loss has a small bump at epoch 10 (0.00023, against about 0.0001 before). This is normal noise from Adam with mini-batches, and it goes down again after.

Because the baseline is already at 0.9999, the GridSearch and the regularization can only change the result by very small amounts.

### 3. Hyperparameter search (GridSearch)

I used `GridSearchCV` from scikit-learn, with the PyTorch models wrapped by `skorch` (`NeuralNetRegressor`). It uses 3-fold cross-validation and the R² score. To keep it fast, I ran it on a random sample of 50,000 training rows (32 combinations × 3 folds = 96 fits). Then I retrained the best configuration on the full training set.

| Hyperparameter | Values tested |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimizer | Adam, SGD |
| Epochs | 10, 20 |
| Neurons per hidden layer | 32, 64 |
| Hidden layers | 1, 2 |

**Best configuration:** Adam, learning rate 0.01, 20 epochs, 2 hidden layers of 64 neurons. Its cross-validated R² is 0.99986, very close to the baseline (0.9999).

This configuration is the biggest one of the grid on every axis (highest learning rate, most epochs, most neurons, most layers). So it sits at the edge of the grid, and a larger grid could give a little more.

### 4. Learning curves (Loss/Epochs and R²/Epochs)

Baseline model:

![Baseline curves](images/curves_base.png)

GridSearch model, retrained on the full training set:

![GridSearch curves](images/curves_best.png)

- For both models, the train and test curves almost overlap. There is no overfitting.
- The baseline reaches a low loss in the first epochs, then moves up and down a little. Its R² stays between 0.99973 and 0.99993. These changes are tiny (about 1e-4).
- The GridSearch model is less stable. Its loss has spikes (up to about 6e-3), its R² falls to 0.994 at epoch 17, and it ends around 0.9992, below the baseline.
- So the "best" model from the search is not better in practice. I see two reasons. First, the search ran on 50,000 rows, where one epoch has far fewer weight updates than on the full training set, so a learning rate of 0.01 that looked fine becomes too high on the full data. Second, the cross-validation score is taken after the last epoch, which is noisy, and the differences between configurations (5th decimal of R²) are as small as this noise.

### 5. Regularization and comparison

I trained seven variants of the same MLP (4 → 64 → 32 → 1, Adam, learning rate 1e-3, 20 epochs) under the same conditions. The scheduler and the early stopping use a validation set (10% of the training set), so the test set is never used to make decisions.

| Model | R² train | R² test | MSE test | Epochs run | Best epoch (val) | Instability* | Time (s) |
|---|---|---|---|---|---|---|---|
| 1. Baseline | 0.999926 | 0.999924 | 0.000077 | 20 | 17 | 0.000013 | 122.7 |
| 2. Weight decay (1e-4) | 0.999813 | 0.999809 | 0.000193 | 20 | 16 | 0.000035 | 123.3 |
| 3. Dropout (0.1) | 0.998209 | 0.998212 | 0.001801 | 20 | 1 | 0.000443 | 123.4 |
| 4. BatchNorm | 0.999538 | 0.999533 | 0.000471 | 20 | 13 | 0.000179 | 126.4 |
| 5. LR scheduler (ReduceLROnPlateau) | 0.999927 | 0.999925 | 0.000076 | 20 | 16 | 0.000005 | 121.8 |
| 6. Early stopping (patience 5) | 0.999935 | 0.999934 | 0.000067 | 20 | 17 | 0.000013 | 121.5 |
| 7. Combined (BatchNorm + weight decay + scheduler + early stopping) | 0.999751 | 0.999750 | 0.000252 | 10 | 5 | 0.000297 | 65.7 |

\*Standard deviation of the test R² over the last 5 epochs (lower means more stable).

![Regularization comparison](images/regularisation_comparaison.png)

- There is no overfitting to fix. The gap between train and test R² is at most 5e-6 for every variant.
- Early stopping, the scheduler and the baseline are almost the same (test R² about 0.99992). Early stopping has the lowest test MSE (0.000067) and the scheduler is the most stable (instability 5e-6). The differences are about 1e-5 and come from one run, so I do not consider them significant.
- Early stopping never stopped the training early here (20 epochs were run). Its small gain comes from restoring the weights of the best validation epoch (17) instead of the last epoch.
- The scheduler gives the smoothest curves.
- Dropout, BatchNorm and weight decay made the results worse. My guess, which I did not test: the network is small and the target is almost a direct function of the inputs, so the noise from Dropout and BatchNorm hurts, and weight decay pushes the weights away from the values needed for an almost linear relation.
- The combined model is worse than the baseline. Early stopping triggered at epoch 10 (best epoch 5), probably because the validation loss of the BatchNorm network is noisy. It is the fastest model (65.7 s), but less accurate.

### 6. Conclusions and limits

- A small MLP predicts `close` with R² of about 0.9999. The problem is easy because `close` is almost fixed by `open`, `low` and `high`.
- Here, regularization improves stability (scheduler) and model choice (early stopping with best weights), not generalization.
- The hyperparameters chosen on a sample did not carry over to the training on the full data.
- Limits: random split instead of a split by date, one random seed, a small grid with the best point on its edge, and R² used in place of accuracy.

## Part 2: Multi-class classification (Predictive Maintenance)

**Dataset:** [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification). It has 10,000 records of a machine process (product type, air temperature, process temperature, rotational speed, torque, tool wear) and two label columns: `Target` (failure yes or no) and `Failure Type` (the type of failure).

**Goal:** predict `Failure Type` (6 classes) from the sensor values.

### 1. Pre-processing

**Checks**

- There are no missing values, no duplicate rows and no duplicate `Product ID`.
- Comparing `Target` and `Failure Type` showed 9 rows with `Target = 1` but `Failure Type = No Failure`. These labels contradict each other, so I removed them. 9,991 rows are left.
- The 18 `Random Failures` rows have `Target = 0`. I kept them as their own class, but I expect them to be very hard to predict.

**What I did**

| Step | What I did | Why |
|---|---|---|
| Columns `UDI` and `Product ID` | Removed | They are only identifiers |
| Column `Target` | Removed | It comes from `Failure Type`, so keeping it would be data leakage |
| Column `Type` (L, M, H) | Ordinal encoding: L=0, M=1, H=2 | Low, medium and high quality have a natural order |
| Target `Failure Type` | `LabelEncoder`: 0 = Heat Dissipation, 1 = No Failure, 2 = Overstrain, 3 = Power, 4 = Random, 5 = Tool Wear | The model needs numbers |
| Train/test split | Stratified 80/20 (`random_state=42`): 7,992 and 1,999 samples | Each class keeps its share in both sets |
| Standardization | `StandardScaler`, fitted on the training set only | The variables have very different scales (about 300 K, 1500 rpm, 40 Nm, 0 to 250 min) |

I split the data before any balancing. The balancing is applied to the training set only, so the test set keeps the real distribution.

### 2. Exploratory data analysis

**Class distribution**

| Failure Type | Label | Samples | Share |
|---|---|---|---|
| No Failure | 1 | 9,643 | 96.52% |
| Heat Dissipation Failure | 0 | 112 | 1.12% |
| Power Failure | 3 | 95 | 0.95% |
| Overstrain Failure | 2 | 78 | 0.78% |
| Tool Wear Failure | 5 | 45 | 0.45% |
| Random Failures | 4 | 18 | 0.18% |

![Class distribution](images/p2_class_distribution.png)

The data is extremely imbalanced. A model that always answers "No Failure" would get 96.5% accuracy and detect nothing. So accuracy alone is misleading. I use the recall and F1-score of each class, and the macro F1.

**Variables by failure type**

![Boxplots by failure type](images/p2_boxplots_by_failure.png)

Mean of each variable for each class:

| Failure Type | air_temp | process_temp | rot_speed | torque | tool_wear |
|---|---|---|---|---|---|
| Heat Dissipation Failure | 302.6 | 310.8 | 1338.0 | 52.8 | 107.3 |
| No Failure | 300.0 | 310.0 | 1540.4 | 39.6 | 106.7 |
| Overstrain Failure | 299.9 | 310.1 | 1354.2 | 56.9 | 208.2 |
| Power Failure | 300.1 | 310.0 | 1764.0 | 48.5 | 101.9 |
| Random Failures | 300.8 | 310.8 | 1489.4 | 43.5 | 119.9 |
| Tool Wear Failure | 300.3 | 310.2 | 1570.7 | 37.2 | 216.6 |

- Tool Wear and Overstrain failures have a high tool wear (mean about 217 and 208 min, against about 107 for normal machines) with a narrow spread.
- Overstrain also has a high torque (56.9 Nm) and a low speed.
- Heat Dissipation has a higher air temperature (302.6 K against 300.0 K) and a low speed (about 1,340 rpm).
- Power failures have a very wide spread of speed and torque, with extreme values on both sides. It looks like two different kinds of situations.
- Random failures look like normal operation on every variable.
- The process temperature hardly changes between classes.

**Torque and rotational speed**

![Torque vs rotational speed](images/p2_torque_vs_speed.png)

- Torque goes down when speed goes up, along a curved line. This looks like a roughly constant power in normal operation.
- Power failures are at both ends of this curve: high torque with low speed (top left), and very low torque with high speed (above about 2,150 rpm, bottom right). Maybe the mechanical power is too high or too low in these cases.
- Overstrain and Heat Dissipation failures form small groups at low speed (about 1,250 to 1,450 rpm) with medium to high torque.
- Tool Wear and Random failures lie on the normal curve, so torque and speed cannot separate them from normal operation. For Tool Wear, the tool wear variable is what makes the difference.

**Correlations**

![Correlation matrix](images/p2_correlation.png)

`rot_speed` and `torque` are strongly negatively correlated, which matches the scatter plot above.

**What this means for the model**

1. The training set must be balanced (see section 3), and I must judge the model with the recall and F1 of each class, the macro F1 and a confusion matrix.
2. Some classes overlap with normal operation (Tool Wear, Random) and the relations are not linear (the torque and speed curve, failures at both ends). So a non-linear model like an MLP makes sense.
3. I expect `Random Failures` to stay unpredictable, whatever the model.
4. The test set has very few samples of the rarest classes (4 `Random Failures` and 9 `Tool Wear Failures`), so the metrics for these classes will be noisy.
5. An idea I did not use in this lab: add new variables such as the temperature difference, the mechanical power (torque × speed) or torque × tool wear.

### 3. Data augmentation (class balancing)

I balanced the **training set only**. The test set keeps its real, imbalanced distribution.

- I reduced `No Failure` to 2,000 samples with `RandomUnderSampler`.
- I increased each other class to 2,000 samples with **SMOTENC**. Plain SMOTE would mix values of the category `Type` and create values that make no sense (like 0.37). SMOTENC treats `Type` as a category and always gives a valid value (0, 1 or 2).
- I used both methods together because raising every class to about 7,700 samples would create thousands of fake points from only 14 real `Random Failures`.

| Class | Before | After |
|---|---|---|
| Heat Dissipation Failure | 90 | 2,000 |
| No Failure | 7,714 | 2,000 |
| Overstrain Failure | 62 | 2,000 |
| Power Failure | 76 | 2,000 |
| Random Failures | 14 | 2,000 |
| Tool Wear Failure | 36 | 2,000 |

![Class balancing](images/p2_balancing.png)

The fake samples of the rarest classes are only mixes of very few real points (14 for `Random Failures`), so they bring little new information. Since `Random Failures` look like normal operation, I still expect the model to detect them badly.

### 4. MLP architecture

- **Input:** the 6 standardized variables. **Output:** 6 values, one per failure type, with `CrossEntropyLoss` (the softmax is inside the loss).
- **Layers:** 6 → 64 → 32 → 6, with ReLU.
- **Training:** Adam, learning rate 1e-3, 30 epochs, batch size 256, on the balanced training set (12,000 samples). After each epoch I also evaluate on the real test set (1,999 samples).

### 5. Hyperparameter search (GridSearch)

I used `GridSearchCV` from scikit-learn with `skorch` (`NeuralNetClassifier`).

A normal cross-validation on the balanced training set would be wrong. Fake points made from the same real sample could end up in both the training fold and the validation fold, and the scores would look too good. So I did this:
- I split the real training set 80/20 (stratified).
- I balanced the 80% part with the same method as in section 3 (12,000 samples).
- I kept the 20% part (1,599 samples) untouched, with the real distribution, as the validation set.
- A `PredefinedSplit` makes `GridSearchCV` use exactly this split.

The score is the macro F1, because accuracy would reward a model that always predicts "No Failure".

| Hyperparameter | Values tested |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimizer | Adam, SGD |
| Epochs | 20, 40 |
| Neurons per hidden layer | 32, 64 |
| Hidden layers | 1, 2 |

**Best configuration:** Adam, learning rate 0.01, 40 epochs, 1 hidden layer of 64 neurons (validation macro F1 = 0.345).

| lr | Epochs | Neurons | Layers | Optimizer | Validation macro F1 |
|---|---|---|---|---|---|
| 0.01 | 40 | 64 | 1 | Adam | 0.345 |
| 0.001 | 40 | 64 | 2 | Adam | 0.336 |
| 0.01 | 40 | 32 | 2 | Adam | 0.331 |
| 0.01 | 40 | 32 | 1 | Adam | 0.330 |
| 0.01 | 20 | 64 | 1 | Adam | 0.316 |

- The macro F1 is low, and I think this is because the validation set keeps the real distribution (about 96.5% `No Failure`) while the model learns on balanced data. The model raises false alarms on healthy machines, which lowers the precision of the small classes. Also, `Random Failures` have only about 3 samples in the validation set and are almost impossible to predict. I checked the details per class in section 7.
- The best configurations are close to each other. They come from one validation split with very few rare samples, so I would not read too much into the order.
- The best configuration is on the edge of the grid (highest learning rate, 40 epochs, and the top 4 all use 40 epochs). A larger grid could give a little more.
- Unlike in Part 1, the search used training data of the same size as the final training (12,000 samples), so the chosen settings should carry over better.

### 6. Learning curves

I plotted the curves for the base MLP and for the GridSearch MLP. The training curve is computed on the balanced training set and the test curve on the real test set, so the two are not directly comparable.

![Base MLP and GridSearch MLP curves](images/p2_curves_base.png)

![GridSearch MLP curves](images/p2_curves_best.png)

Base MLP (30 epochs):

| Epoch | Train loss | Test loss | Train accuracy | Test accuracy |
|---|---|---|---|---|
| 1 | 1.3520 | 1.5318 | 0.6027 | 0.3087 |
| 5 | 0.3514 | 0.8183 | 0.8888 | 0.6288 |
| 10 | 0.2359 | 0.6669 | 0.9248 | 0.6998 |
| 15 | 0.1915 | 0.5453 | 0.9407 | 0.7629 |
| 20 | 0.1650 | 0.5794 | 0.9517 | 0.7544 |
| 25 | 0.1441 | 0.4793 | 0.9624 | 0.8014 |
| 30 | 0.1311 | 0.4957 | 0.9633 | 0.8014 |

GridSearch MLP (40 epochs):

| Epoch | Train loss | Test loss | Train accuracy | Test accuracy |
|---|---|---|---|---|
| 1 | 0.4037 | 0.9036 | 0.8531 | 0.6063 |
| 5 | 0.1861 | 0.5964 | 0.9429 | 0.7319 |
| 10 | 0.1391 | 0.5157 | 0.9579 | 0.7974 |
| 15 | 0.1110 | 0.4760 | 0.9672 | 0.8304 |
| 20 | 0.0943 | 0.4432 | 0.9733 | 0.8419 |
| 25 | 0.0895 | 0.4953 | 0.9745 | 0.8424 |
| 30 | 0.0811 | 0.3417 | 0.9817 | 0.8889 |
| 35 | 0.0748 | 0.3439 | 0.9834 | 0.8969 |
| 40 | 0.0671 | 0.4054 | 0.9822 | 0.8834 |

- **Base MLP:** the training loss goes down from 1.35 to 0.13 and the training accuracy goes up from 0.60 to 0.96. The test loss goes down more slowly (from 1.53 to about 0.50) and the test accuracy stays between 0.79 and 0.82 from epoch 22. It learns more slowly than the GridSearch MLP, probably because its learning rate is 10 times smaller (1e-3 against 1e-2), and it ends lower (0.80 against 0.88).
- **GridSearch MLP:** the training loss goes down steadily and the training accuracy reaches about 98%. The test loss stays 5 to 6 times higher (0.34 to 0.50) and moves up and down from one epoch to the next. The test accuracy goes from 0.61 to about 0.89, is still improving until epoch 35, then drops a little at epoch 40. This fits the fact that 40 epochs is the edge of the grid, so the model could still learn more.
- For both models there is a clear gap between training and test. Part of it comes from the change of distribution (balanced for training, imbalanced for testing), and part comes from overfitting on the rare classes (see section 7).

### 7. Metrics

I computed the metrics for both models on three sets: the balanced training set (what the model saw), the training set with its real distribution, and the real test set. *Sensitivity* is the recall, meaning the share of the real failures of a class that the model finds.

| Model | Set | Accuracy | Precision (macro) | Sensitivity (macro recall) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|---|
| Base MLP | Train (balanced) | 0.9632 | 0.9651 | 0.9632 | 0.9623 | 0.9623 |
| Base MLP | Train (real distribution) | 0.8197 | 0.4077 | 0.9054 | 0.4855 | 0.8844 |
| Base MLP | Test (real distribution) | 0.8014 | 0.3941 | 0.6813 | 0.4625 | 0.8730 |
| GridSearch MLP | Train (balanced) | 0.9822 | 0.9826 | 0.9822 | 0.9819 | 0.9819 |
| GridSearch MLP | Train (real distribution) | 0.8946 | 0.4484 | 0.9284 | 0.5377 | 0.9301 |
| GridSearch MLP | Test (real distribution) | 0.8834 | 0.4073 | 0.6487 | 0.4797 | 0.9224 |

Results of the GridSearch MLP for each class on the test set:

| Failure Type | Precision | Recall (sensitivity) | F1-score | Support |
|---|---|---|---|---|
| Heat Dissipation Failure | 0.458 | 1.000 | 0.629 | 22 |
| No Failure | 0.994 | 0.886 | 0.937 | 1929 |
| Overstrain Failure | 0.533 | 1.000 | 0.696 | 16 |
| Power Failure | 0.447 | 0.895 | 0.596 | 19 |
| Random Failures | 0.000 | 0.000 | 0.000 | 4 |
| Tool Wear Failure | 0.011 | 0.111 | 0.021 | 9 |

![Confusion matrix](images/p2_confusion_matrix.png)

- **The change of distribution is the main problem.** The drop already shows on the real training data (accuracy 0.895, macro precision 0.448) and it is similar on the test set (0.883 and 0.407). The model learns on balanced data, but in real life about 96.5% of the machines are healthy, so the precision of the small classes falls.
- **There is also some overfitting on the rare classes.** The macro recall is 0.928 on the real training set but 0.649 on the test set. The model finds the rare failures it has already seen much better than new ones.
- **Three failure types are found well.** Heat Dissipation, Overstrain and Power have a recall of 1.00, 1.00 and 0.89 (55 of 57 failures found). Their precision is only 0.45 to 0.53 because of false alarms.
- **There are many false alarms.** 219 of the 1,929 healthy machines (11.4%) are flagged as failing. This is why the accuracy (88.3%) is lower than the 96.5% of a model that always says "No Failure". The model gives up some accuracy to detect failures.
- **Tool Wear and Random Failures are not found.** Their recall is 0.11 and 0.00. 87 healthy machines are predicted as Tool Wear and 74 as Random. I think this happens because these classes look like normal operation (see the EDA), so their thousands of fake samples, made from only 36 and 14 real points, cover the area of normal machines.
- **GridSearch MLP against base MLP.** The GridSearch MLP has a better accuracy (0.883 against 0.801) and a slightly better macro F1 (0.480 against 0.463), with a slightly lower macro recall (0.649 against 0.681). The gain is small and comes from one run.
- **The numbers for the rare classes are noisy.** The test set has only 4 `Random Failures` and 9 `Tool Wear Failures`. One more correct prediction changes the recall of Tool Wear by 11 points.
- **Ideas I did not test:** keep more `No Failure` samples when balancing, use class weights in the loss, change the decision thresholds, or add new variables (temperature difference, mechanical power, torque × tool wear).

### 8. Regularization and comparison

I trained six variants of the best architecture from the GridSearch (6 → 64 → 6, Adam, learning rate 0.01, 40 epochs) under the same conditions:

- **Training data:** the balanced "fit" part of the training set (12,000 samples), without the validation samples.
- **Validation:** the real validation set (1,599 samples, never used for training), used to choose the best epoch with the macro F1.
- **Test:** the real test set, used only for the final report.

I retrained the baseline in this setup, so its numbers are different from section 7, where the model was trained on the whole balanced training set.

| Model | Train acc (balanced) | Best val macro F1* | Test acc | Test macro precision | Test macro recall | Test macro F1 | Epochs run | Best epoch (val) | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1. Baseline | 0.9877 | 0.5512 | 0.9135 | 0.4265 | 0.6278 | 0.4932 | 40 | 33 | 4.9 |
| 2. Weight decay (1e-3) | 0.9680 | 0.4575 | 0.8249 | 0.3452 | 0.6484 | 0.4119 | 40 | 39 | 4.9 |
| 3. Dropout (0.3) | 0.9814 | 0.5047 | 0.8924 | 0.4049 | 0.6687 | 0.4806 | 40 | 32 | 5.3 |
| 4. BatchNorm | 0.9812 | 0.5346 | 0.8934 | 0.4060 | 0.6146 | 0.4728 | 40 | 33 | 5.4 |
| 5. Early stopping (patience 8) | 0.9808 | 0.5263 | 0.8969 | 0.4426 | 0.6250 | 0.5023 | 29 | 21 | 3.6 |
| 6. Combined (Dropout 0.3 + weight decay 1e-4 + early stopping) | 0.9760 | 0.4689 | 0.8724 | 0.3886 | 0.6381 | 0.4589 | 31 | 23 | 4.0 |

\*Highest validation macro F1 reached during the training. For the runs without early stopping, the model that is evaluated is the one from the last epoch.

![Regularization comparison](images/p2_regularization_comparison.png)

The baseline and the early stopping run use the same seed, so they follow the same path until early stopping triggers at epoch 29. This is why the baseline curve is hidden behind the other one.

- **Early stopping** gives the best test macro F1 (0.502) and the best macro precision (0.443), and it needs fewer epochs (29 instead of 40). The gain over the baseline (+0.009 in macro F1) is too small to count as real.
- **Weight decay (1e-3)** clearly makes things worse (test macro F1 0.412, accuracy 0.825, and a lower training accuracy). The penalty is probably too strong for this small network, so it underfits. I did not try other values.
- **Dropout (0.3)** gives the highest macro recall (0.669) but a lower accuracy and precision. It finds more failures but raises more false alarms. Its macro F1 (0.481) is a little under the baseline.
- **BatchNorm** gives the noisiest validation curve and no gain (macro F1 0.473).
- **The combined model** (macro F1 0.459) is worse than each of its parts alone. Using several techniques together did not add up.
- **No technique clearly beats the baseline.** All of them lower the training accuracy a little (from 0.988 to 0.976 to 0.981) without improving the test results. The gap between training and test comes mostly from the change of distribution and from the very small number of rare samples (see section 7), and these general techniques do not fix that.
- **There is a lot of noise.** The same configuration gets a test accuracy of 0.883 when trained on the whole balanced training set (section 7) and 0.914 here. This is a 3-point difference without any regularization, as big as most differences between the techniques. The test set has only 4 `Random Failures` and 9 `Tool Wear Failures`, the validation macro F1 moves by several points from one epoch to the next (the validation set has only about 3 and 7 samples of these classes), and I used one seed. Only the big effect (weight decay at 1e-3 hurts) is reliable. To rank the other techniques, I would need several seeds or cross-validation.

### 9. Conclusions and limits

- I built the pipeline carefully: I removed the contradicting labels and the `Target` column that leaks the answer, I split the data before balancing, I fitted the scaler on the training set only, and I balanced the training data only (under-sampling and SMOTENC).
- The final MLP has a test accuracy of about 0.88 to 0.91 and a macro F1 of about 0.48 to 0.50. It finds three failure types well (recall of at least 0.89 for Heat Dissipation, Overstrain and Power) but with many false alarms (about 11% of the healthy machines are flagged). It does not find Tool Wear and Random Failures.
- Regularization did not give a clear gain. Early stopping was the most useful, with the same performance in fewer epochs.
- **Limits:** the rarest classes have very few samples (14 `Random Failures` and 36 `Tool Wear Failures` in the training set, 4 and 9 in the test set), so their metrics are noisy. I used one random seed. Balancing to 2,000 samples per class is probably too strong for the rarest classes. I did not add new variables or tune the decision thresholds.
- **What I would try next:** a lighter balancing, class weights in the loss, new variables (temperature difference, mechanical power, torque × tool wear), tuning the thresholds, and several runs or cross-validation to compare techniques fairly.

## What I learned

- How to build and train an MLP in PyTorch: tensors, `Dataset` and `DataLoader`, the training loop and the evaluation loop, for both regression and classification.
- How to avoid data leakage: fit the scaler on the training data only, split before balancing, remove a column that comes from the target, and keep a validation set so the test set is never used to make decisions.
- How to run a hyperparameter search with scikit-learn and skorch, and what can go wrong: a search on a small sample may not work on the full data, a cross-validation on fake samples leaks information (a `PredefinedSplit` with a real validation set fixes this), and noisy scores make small differences meaningless.
- Accuracy can be misleading with imbalanced data. The recall of each class, the macro F1 and the confusion matrix show what really happens. Balancing the training set changes the class distribution, so the model finds more rare failures but raises more false alarms.
- How to read learning curves (overfitting, convergence, noise) and how to pick a good scale for the axes.
- Regularization does not always help. It only helps if the model really overfits, and with very few samples per class and one run, small differences between techniques cannot be trusted.

## How to run

1. Open the notebooks on Kaggle and add the two datasets as inputs: [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse) and [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).
2. Turn on Internet in the notebook settings (needed to install `skorch`).
3. Run all the cells from top to bottom.
