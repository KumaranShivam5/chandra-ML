# Imports


```python
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 
%load_ext autoreload
%autoreload 2
```

# Load Training data


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

### Components for the _make_model_ object

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


```python

model = make_model(model_name = 'test_model', classifier=clf, oversampler = oversampler, train_data = x, label=y)

# Validate model
model.validate(save_predictions=True, multiprocessing=True, k_fold=20)

# Print validation result
print("Confusion Matrix: ")
print(model.validation_score['class_labels'])
print(model.validation_score['confusion_matrix'])
print("Overall Scores: ")
print(model.validation_score['overall_scores'])
print("Class-Wise scores: ")
print(model.validation_score['class_wise_scores'])

# Once satisfied with the mdoel performance
# train the mdoel on entire training dataset
model.train()

# save the mdoel
model.save('model_filename.joblib')
```
