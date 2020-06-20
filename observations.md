# Observations

### KNN

#### Imbalanced classes

Over some testings: weighted f1-score: most times > 36% ( 30%- 40 %)

```python
              precision    recall  f1-score   support

     Neutral       0.34      0.33      0.34        33
       Anger       0.54      0.56      0.55        45
        Fear       0.24      0.21      0.22        19
         Joy       0.21      0.26      0.23        19
     Sadness       0.50      0.50      0.50        18
     Disgust       0.18      0.11      0.14        18
     Boredom       0.38      0.44      0.41        25

    accuracy                           0.38       177
   macro avg       0.34      0.34      0.34       177
weighted avg       0.37      0.38      0.37       177
```

and the confusion matrix:

<img src="./plotting/plots/knn_unbalanced.png" alt="knn_unbalanced" style="zoom:50%;" />

Due to the fact that some classes appear more than others in out training set (`X_train`), I try oversampling.

<img src="./plotting/plots/class_stats.png" alt="class_stats" style="zoom:50%;"/>

So I try to oversample classes using `imblearn.over_sampling` and check new results.

#### Oversampling

Over some testings: weighted average: more than 37% (35% - 41%)

```python
              precision    recall  f1-score   support

     Neutral       0.23      0.50      0.32        20
       Anger       0.66      0.57      0.61        44
        Fear       0.23      0.20      0.21        25
         Joy       0.43      0.33      0.38        27
     Sadness       0.58      0.55      0.56        20
     Disgust       0.15      0.20      0.17        15
     Boredom       0.57      0.31      0.40        26

    accuracy                           0.40       177
   macro avg       0.41      0.38      0.38       177
weighted avg       0.45      0.40      0.41       177

```

and the confusion matrix down below:



<img src="./plotting/plots/knn_balanced_SMOTE.png" alt="knn_balanced_SMOTE" style="zoom:50%;" />

### MLP

In the multilayer perceptron the hidden size was set to 50.

#### Imbalanced classes 

```python
              precision    recall  f1-score   support

     Neutral       0.31      0.19      0.24        21
       Anger       0.65      0.66      0.65        50
        Fear       0.15      0.14      0.15        21
         Joy       0.32      0.30      0.31        23
     Sadness       0.69      0.69      0.69        16
     Disgust       0.77      0.45      0.57        22
     Boredom       0.36      0.62      0.45        24

    accuracy                           0.47       177
   macro avg       0.46      0.44      0.44       177
weighted avg       0.48      0.47      0.47       177

```

And the confusion matrix:

<img src="./plotting/plots/mlp_unbalanced_1.png" alt="mlp_unbalanced_1" style="zoom:50%;" />



#### Oversampling

Using `sklearn.neural_network.MLP_CLASSIFIER` performance actually boosted up to 53%.
```python
              precision    recall  f1-score   support

     Neutral       0.71      0.36      0.48        28
       Anger       0.60      0.60      0.60        42
        Fear       0.43      0.38      0.41        26
         Joy       0.31      0.50      0.38        28
     Sadness       0.76      0.73      0.74        22
     Disgust       0.62      0.71      0.67        14
     Boredom       0.56      0.53      0.55        17

    accuracy                           0.53       177
   macro avg       0.57      0.54      0.55       177
weighted avg       0.57      0.53      0.53       177
```
The confusion matrix is shown below:

<img src="./plotting/plots/mlp_balanced_SMOTE_1.png" alt="mlp_balanced_SMOTE_1" style="zoom:50%;" />

#### PCA

Using Principal Component Analysis model's weighted average f1-score seems to be more stable (most times over 40%) but lower.

```python
              precision    recall  f1-score   support

     Neutral       0.50      0.26      0.34        31
       Anger       0.60      0.55      0.57        44
        Fear       0.37      0.62      0.46        21
         Joy       0.41      0.57      0.47        23
     Sadness       0.53      0.53      0.53        19
     Disgust       0.21      0.25      0.23        16
     Boredom       0.38      0.26      0.31        23

    accuracy                           0.44       177
   macro avg       0.43      0.43      0.42       177
weighted avg       0.46      0.44      0.44       177
```

### SVM

#### CV

Using 7-fold cross-validation, RBF SVM scored 53.24 % .

#### Oversampling

I used the implementation of SVM from `sklearn.svm.NuSVC`  (~50%)

```python
              precision    recall  f1-score   support

     Neutral       0.46      0.48      0.47        27
       Anger       0.77      0.53      0.63        32
        Fear       0.39      0.44      0.41        16
         Joy       0.45      0.42      0.43        24
     Sadness       0.65      0.71      0.68        21
     Disgust       0.50      0.50      0.50        14
     Boredom       0.53      0.67      0.59        27

    accuracy                           0.54       161
   macro avg       0.54      0.54      0.53       161
weighted avg       0.56      0.54      0.54       161
```



<img src="./plotting/plots/svm_balanced_2.png" alt="svm_balanced_2" style="zoom:50%;" />

From all the above better performance was achieved by **SVMusing OverSampling**  .



### LSTM

Long short term memory Neural Networks should perform best as guested in prior.

The following network used SGD and`learning_rate=0.1` but no regularization techniques were used.

It was trained for 30 epochs.

```
Accuracy on validation set: 28.57142857142857 %
Accuracy on test set: 57.14285714285714 %
```



Using `Adam optimizer` 

```
Accuracy on validation set: 42.857142857142854 %
Accuracy on test set: 42.857142857142854 %
```

#### RandomOversampler

Train for `30 epochs`.

```
Accuracy on validation set: 57.14285714285714 %
Accuracy on test set: 42.857142857142854 %
f1 score: 0.41099427224959345
```

Now we try for more epochs to check for overfit. => Model is not complex enough.

```
Accuracy on validation set: 35.714285714285715 %
Accuracy on test set: 35.714285714285715 %
f1 score: 0.20681785629028404
```

So we now check a more complex model.

##### `epochs=50, dropout=0.1, n_layers=1`

```
Validation loss at epoch 45 : 8.98932933807373
Train loss at epoch 49 : 59.959686279296875
Accuracy on validation set: 28.57142857142857 %
Accuracy on test set: 21.428571428571427 %
f1 score: 0.28614808723652263
```

##### `epochs=100, dropout=0.1, n_layers=1`

```
Validation loss at epoch 95 : 8.386117935180664
Train loss at epoch 99 : 56.79729461669922
Accuracy on validation set: 28.57142857142857 %
Accuracy on test set: 35.714285714285715 %
f1 score: 0.3100350061133705
```



##### `epochs=100, dropout=0.2, n_layers=1`

```
Validation loss at epoch 95 : 8.8601655960083
Train loss at epoch 99 : 60.11253356933594
Accuracy on validation set: 50.0 %
Accuracy on test set: 42.857142857142854 %
f1 score: 0.2832810785132766
```

