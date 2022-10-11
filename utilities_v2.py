"""

This code implements a custom version of K-fold cross-validation: 
Cumulative K-fold cross-validation.
Training, Validation, and predictions are encapsulated in the class: make_model
This class allows the user to pass his choice of classifier 
and oversampler to the pipeline and train and validate the model for their dataset.
For training and validation over large datset the user can choose multiprocessing.  
This code aims to make the model selection and tuning simple and efficient.


Author : Shivam Kumaran
"""

from tqdm import tqdm 
import numpy as np 
import pandas as pd 

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold 

def deets(df ,class_info = 0 ,dsp=0):
    print('_____________________________________________________')
    if(dsp):
        display(df)
        print('_____________________________________________________')
    print('------------------------------')
    print(f'Number of Objects : {df.shape[0]}')
    print(f'Number of Columns : {df.shape[1]}')
    if(class_info):
        print('------------------------------')
        display(df['class'].value_counts())
    print('_____________________________________________________')
#df_deets(data)


def train_classifier_model(arr):

    """
    For a sample size of N, and given indices of train and validation data, performs training on the train-data and does predictions on the validation data

    Parameters
    ----------
    arr : array
        Should contain : [clf, oversampler,data, label, index]
            clf : sklearn classifier model which implements fit, predict and predict_proba methods
            oversampler : oversampleing object, default = None
                This oversampling model must implement fit_resample method. Enables oversampling to mitigate class imbalance issue. Give none for no oversampling
            data : training data
            label : labels for training data
            index : array of length 2 : [training_indices, test_indices]

    Returns
    -------
    df : dataframe
        columns : 
            true_class : true class
            pred_class : predicted class
            pred_prob : membership probability for the predicted class
            prob_<class> : membership probability of all classes
    """

    clf,oversampler,x,y, index = arr
    train_index, test_index, = index[0], index[1]
    x_train, x_test = x.iloc[train_index, : ], x.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # If oversampler is provided, then the training data is oversampled
    # and then the model is trained on the upsampled dataset.
    if(oversampler):
        x_train_up, y_train_up = oversampler.fit_resample(x_train, y_train)
        # x_train_up, y_train_up = oversampling(oversampler , x_train, y_train)
        # x_train_up = x_train_up.replace(np.nan , -100)
        clf.fit(x_train_up, y_train_up)
    else: 
        clf.fit(x_train, y_train)

    # Putting the prediction results: predicted class and the membership probability
    # in a DataFrame
    df = pd.DataFrame({
        'name' : x_test.index.to_list(), 
        'true_class' : y_test, 
        'pred_class' : clf.predict(x_test), 
        'pred_prob' : [np.amax(el) for el in clf.predict_proba(x_test)]
    }).set_index('name')

    # Class membership probabilities for all classes are 
    # arranged in tabular form and is appended to the prediction dataframe
    membership_table = pd.DataFrame(clf.predict_proba(x_test), columns=[f'prob_{el}' for el in clf.classes_])
    membership_table.insert(0, 'name', x_test.index.to_list())
    membership_table = membership_table.set_index('name')
    df = pd.merge(df, membership_table, left_index=True, right_index=True)

    return df




def cumulative_cross_validation(x,y, classifier, oversampler = None,  k_fold=-1, multiprocessing = True ):

    """
    Performs a cumulative cross validation 
    In standard K-fold of Leave one out cross validation, 
    model scores are calculated on each fold and then the average of scores are reported. 
    In this custom version of validation, we accumulate the predictions from each folds, and then calculate the model scores.

    Parameters
    ----------
    x : Pandas Dataframe
        training data of size (N,M), N is number of samples, M is number of features
    y : Pandas Series
        Training Labels of size N
    classifier : sklearn model
        Classifier model. This mode must implement fit, predict and predict_proba methods
    oversampler : oversampleing object, default = None
        This oversampling model must implement fit_resample method. Enables oversampling to mitigate class imbalance issue. Give none for no oversampling
    k_fold : int, default=-1
        Number of folds for cross validation. -1 for leave one out cross vlidation
    multiprocessing : Boolean, default=True
        Select if cross validation is performed with multiprocessing or not.

    Returns
    -------
    DataFrame 
        columns : 
            true_class : true class
            pred_class : predicted class
            pred_prob : membership probability for the predicted class
            prob_<class> : membership probability of all classes (not available fo LeaveOneOut CV)

    """

    # Selection of Cross validation method, based on k_fold value
    if k_fold==-1:
        print('[INFO] >>> Doing LeaveOneOut cross-validation')
        cv = LeaveOneOut()
    else:
        print(f'[INFO] >>> Doing {k_fold} fold cross-validation')
        cv = StratifiedKFold(k_fold)# KFold(k) 

    # Using CV, split indices for training and validation set
    # and creating a list of k_fold elements, with model, data and labels and corresponding indices.
    index = [(t,i) for t,i in cv.split(x,y)]
    zipped_arr = list(zip([classifier]*len(index), [oversampler]*len(index), [x]*len(index), [y]*len(index), index ))
    
    # Training and validation 
    # depending on multiprocessing selected or not.
    if(multiprocessing):
        import multiprocessing as mp 
        num_cores = mp.cpu_count() # selecting all available CPU cores
        print(f"[INFO] >>> Using {num_cores} CPU cores")
        with mp.Pool(int(num_cores)) as pool:
            result = pool.map(train_classifier_model, zipped_arr) 
    else:
        result = []
        print('[INFO] Not using Multi-processing')
        for a in tqdm(zipped_arr):
            result.append(train_classifier_model(a))

    # create dataframe from each fold's prediction dataframes
    result_df = pd.concat(result, axis=0)
    
    return result_df




def get_validation_score(pred_table, confidance=0, score_average_type = 'weighted'):
    """
    Function to calculate various scores from the true labels, predicted labels and predicted probabilities using sklearn's metrics. Both overall scores and class-wise scores are calculated.

    Parameters
    ----------
    pred_table : Dataframe
        Should have the columns :
            true_class : true class labels
            pred_class : predicted class labels
            pred_prob : class membership probability for the predicted class
    confidance : float, range : [0,1]
        probability confidance threshold. Validation scores are calculated only for the samples for which class membership probability is more than the confidance selected.
    score_average_type : string {'weighted', None, 'micro', 'macro'}, default='weighted'
        Choose the averageing method for calcuation of overall precision, recall and f1 score.

    Returns
    -------
    dict : 
        keys:
            'class_labels' 
            'confusion_matrix' 
            'overall_scores': dict
                            keys : 
                                'balanced_accuracy', 
                                'accuracy',
                                'precision', 
                                'recall',
                                'f1',
                                'mcc',
            'class_wise_scores' : DataFrame
                            columns : 
                                'class'
                                'recall_score'
                                'precision_score' 
                                'f1_score'

    """

    # We select only those samples where 
    # class membership probbility is more than given confidance
    pred_table = pred_table[pred_table['pred_prob']>confidance]
    y_true = pred_table['true_class']
    y_pred = pred_table['pred_class']
    labels = np.sort(y_true.unique())
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels )

    score_dict = {
        'class_labels' : list(labels),
        'confusion_matrix' : cm,
        'overall_scores': {
            'balanced_accuracy' : balanced_accuracy_score(y_true, y_pred ), 
            'accuracy' : accuracy_score(y_true, y_pred, ), 
            'precision' : precision_score(y_true, y_pred, average=score_average_type), 
            'recall' : recall_score(y_true, y_pred, average=score_average_type), 
            'f1' : f1_score(y_true, y_pred, average=score_average_type), 
            'mcc' : matthews_corrcoef(y_true, y_pred),
        }, 
        'class_wise_scores' : pd.DataFrame({
            'class' : labels, 
            'recall_score' : recall_score(y_true, y_pred, average=None, ), 
            'precision_score' : precision_score(y_true, y_pred, average=None, ),
            'f1_score' : f1_score(y_true, y_pred, average=None, )
        }).sort_values(by='class').set_index('class'), 
    }
    return score_dict




class make_model():
    """
    From a given classifier, training data and label creates classifier and stores the validation results

    Attributes
    ----------
    name : str
        name of the model
    classifierf : sklearn classifier model object
        Model to be used for classification. Only those models which implement fit, predict and predict_proba methods are allowed
    oversampler : oversampleing object, default = None
        This oversampling model must implement fit_resample method. Enables oversampling to mitigate class imbalance issue. Give none for no oversampling
    train_data : Datarane
         training data without label 
    label : Series
        Labels corresponding to the training data 
    
    Methods
    -------
    validate(fname= '', k=10, normalize_prob=0, score_average = 'macro', save_predictions = '', multiprocessing = True)
        Do the cumulative cross validation on the model, generated predictions on the training set and calculates validation scores
    train()
        train the classifier model in the entire training dataset
    save(fname)
        save the mdoel object with the classifier and validation results.

    """

    def __init__(self, model_name, classifier, oversampler, train_data,label):
        self.name = model_name 
        self.clf = classifier 
        self.train_data = train_data
        self.oversampler = oversampler
        self.label = label
        self.validation_prediction = 'validation predictions are not stored'
        
    def validate(self, k_fold=10, save_predictions = False, multiprocessing = True, score_average_type= 'weighted'):
        """
        Do the cumulative cross validation on the model, generated predictions on the training set and calculates validation scores

        Parameters
        ----------
        k_fold :  int, default = -1
            number of folds to be used for k-fold cross validation. Use k_fold=-1 for LeaveOneOut cross validation
        save_predictions : Boolean
            From the cumulative prediction, we get class membership probabilities for all the classes, which can be stroed for future use as a part of model object itsels by setting the value 'True'
        multi_processing : Boolean
            'True' will use multiprocessing and use all available CPU cores for cross validation.
        score_average_type : string {'weighted', None, 'micro', 'macro'}, default='weighted'
        Choose the averageing method for calcuation of overall precision, recall and f1 score.
        
        """
        validation_predictions = cumulative_cross_validation(self.train_data,self.label,k_fold=k_fold, classifier=self.clf,oversampler = self.oversampler,  multiprocessing=multiprocessing)

        if(save_predictions):
            self.validation_prediction = validation_predictions
        # calclate validation scores
        self.validation_score = get_validation_score(validation_predictions,  score_average_type=score_average_type)
        return self

    def train(self):
        """
        Trains the classifier with the entire training dataset provided.
        """
        clf = self.clf
        clf.fit(self.train_data, self.label)
        return self

    def save(self, fname):
        """
        save the trained model.
        Saves the trained classifier, validation scoress, training data and training labels (and validation predictions table, if selected) as make_model object using joblib module's dump method

        Parameters
        --------
        fname : string
            Relative path with filename where model is to be saved.
        """
        import joblib
        joblib.dump(self, fname , compress=5)





###################################################################

"""

# Example Implementation ####################

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Assume df is the dataframe contains samples with features and class labels
# Class labels are stored in the column names 'class'

# Load data, features are stored in variable x and labels in variable y
df = pd.read_csv('example_data.csv', index_col = 'name')
x = df.drop(columns=['class'])
y = df['class']

# Assume that the above script is given in file named - utilities 
# and in the same working directory

## Uncomment following if the script is used as a module
#from utilities import make_model

# Create a new make_model object
clf = RandomForestClassifier()
oversampler = SMOTE(k_neighbors=2)
model = make_model('test_model', clf, oversampler, x, y)

# Validate model
model.validate(save_predictions=True, multiprocessing=True, k_fold=5)

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


"""