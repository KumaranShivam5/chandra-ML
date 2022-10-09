# Chandra-ML


```python
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 
%load_ext autoreload
%autoreload 2
```

# Data


```python
from utilities import deets
from choices import get_train_data , param_dict
```

Using the function _get_train_data_ in the module _choices_, we load the training data. Using the _id_frame.csv_ files the data is filtered as per the filtering provided as the argument in the _get_train_data_ function


```python
# file = f'../compiled_data_v3/imputed_data_v2/x_phot_minmax_modeimp.csv'
# Select the classes to load
classes = ['AGN' ,'STAR' , 'YSO' ,  'CV' , 'LMXB' , 'HMXB' ,'ULX','PULSAR']

# flag filtering
flag = {
    'conf_flag' : 0 , 
    'streak_src_flag' : 0 , 
    'extent_flag' : 0 , 
    'pileup_flag' : 0 , 
    }
# Load data
data = get_train_data(flags = flag, classes= classes , offset = 1,)

#drop some features
feat_to_drop = param_dict['hardness']+param_dict['IRAC']
data = data.drop(columns = feat_to_drop)

# see data details
deets(df = data,class_info = 0, dsp = 0)
```

    _____________________________________________________
    ------------------------------
    Number of Objects : 7703
    Number of Columns : 42
    _____________________________________________________


# Model Training and Validation

### Import _make_model_ class
The class _make_model_ is takes in the training data, a classification model(scickit-learn compatible model). This class is can be used to validate the model using CCV method and to train and save the classifier for implementation on the test data.


```python
import nbconvert
```


```python
from utilities_v2 import make_model
```

### Build the Model: _make_model_ class

_make_model_ takes in the following components
*   name : user defined name of the model (can be any string)
*   train_data : as pandas dataframe
*   label : class label for the training data (list or pandas series)
*   classifier : classifier model
*   oversamples : Oversampling function like Scickit-Learn's _SMOTE_ object.

#### Data
the class _make_model_ takes in training data and the training label as pandas dataframe


```python
# Example Implementation ####################
x = data.drop(columns=['class'])
y = data['class']
```

#### Classifier

Next we will use a classifier from scickit-learn _RandomForestClassifier_ 

The user can supply their own classifier for the _make_model_ object with only condition that the classifier must implement the _fit_ function. (Need not worry, as most of the models in Scickit-Learn always implement the _fit_ function)

<small>Note: the parameters we are giving for the model that we are giving here is optained after hyper-parameter tuning of the model.</small>


```python
# Create a new make_model object
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=400 , max_depth=30 , random_state=np.random.randint(0,999999))
```

#### Oversampler


```python
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(k_neighbors=4)
```

#### Put everything together 


```python
model = make_model(model_name = 'test_model', classifier=clf, oversampler = oversampler, train_data = x, label=y)
```

### Validate the Model

the object _make_model_ implements *validate* function ehich performs the Cumultive K fold cross validation for the supplied model and for the given data


```python
model.validate(save_predictions=True, multiprocessing=True, k_fold=2)
```

    [INFO] >>> Doing 2 fold cross-validation
    [INFO] >>> Using 8 CPU cores





    <utilities_v2.make_model at 0x7fa1abe20898>



Let us see the validation result

The validation results are stored in the attribute _validation_model_ of the _make_model_ object


```python
# Print validation result
print("Confusion Matrix: ")
print(model.validation_score['class_labels'])
print(model.validation_score['confusion_matrix'])
print("Overall Scores: ")
print(model.validation_score['overall_scores'])
print("Class-Wise scores: ")
print(model.validation_score['class_wise_scores'])
```

    Confusion Matrix: 
    ['AGN', 'CV', 'HMXB', 'LMXB', 'PULSAR', 'STAR', 'ULX', 'YSO']
    [[2197    7  102    0   13   18   56    2]
     [  11   33   29    2   28   42    7   14]
     [  33   15  557    4   11   32   88    8]
     [   4    8    5  106    2    8    4    6]
     [   2   18   17    0   32   11    9   12]
     [  35   34   33    7   19 2572    6   84]
     [  15    6   59    3    7    3  118    0]
     [   2   14    8    4   19   50    2 1050]]
    Overall Scores: 
    {'balanced_accuracy': 0.6642261754119165, 'accuracy': 0.8652473062443204, 'precision': 0.8727987144236677, 'recall': 0.8652473062443204, 'f1': 0.8682408798541683, 'mcc': 0.8190420660929271}
    Class-Wise scores: 
            recall_score  precision_score  f1_score
    class                                          
    AGN         0.917328         0.955633  0.936089
    CV          0.198795         0.244444  0.219269
    HMXB        0.744652         0.687654  0.715019
    LMXB        0.741259         0.841270  0.788104
    PULSAR      0.316832         0.244275  0.275862
    STAR        0.921864         0.940058  0.930872
    ULX         0.559242         0.406897  0.471058
    YSO         0.913838         0.892857  0.903226


### Train the model

Now the above validation function can be used by varying the classifier parameters and then checking the validation result as per the user requirement, and once the results are satisfactoory, the user call the _train_ function of the _make_model_ object which will train and store the supplied classifier. for training, unlike the cross validation where a fraction of th data is used, here the classifier is trained on the entire dataset.


```python
model.train()
```




    <utilities_v2.make_model at 0x7fa1abe20898>



### Save the Model

Next we will use the _save_ function of the object _make_model_ to save the classifier alongwith the validation scores and predictions on the training data


```python
model.save('model_filename.joblib')
```


```python

```
