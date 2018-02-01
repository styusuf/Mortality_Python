import models_partc
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    
    accuracy_scores = []
    auc_scores = []
    
    kf = KFold(X.shape[0], n_folds=5, random_state= 545510477)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # Fit training data to Logistic Regression model and make prediction on test set
        y_pred = models_partc.logistic_regression_pred(X_train, y_train, X_test)
        
        # Compare predictions in y_pred to true values of y_test
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        accuracy_scores.append(acc) #Append acc to list of accuracy score for each fold
        auc_scores.append(auc) #Append auc to list of auc score for each fold
    
    return np.mean(accuracy_scores), np.mean(auc_scores)


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    accuracy_scores = []
    auc_scores = []
    
    ss = ShuffleSplit(X.shape[0], n_iter=iterNo, test_size=test_percent, random_state=545510477)
    
    for train_index, test_index in ss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # Fit training data to Logistic Regression model and make prediction on test set
        y_pred = models_partc.logistic_regression_pred(X_train, y_train, X_test)
        
        # Compare predictions in y_pred to true values of y_test
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        accuracy_scores.append(acc) #Append acc to list of accuracy score for each fold
        auc_scores.append(auc) #Append auc to list of auc score for each fold
        
    
    return np.mean(accuracy_scores), np.mean(auc_scores)


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

