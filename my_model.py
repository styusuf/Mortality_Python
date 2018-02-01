import utils
import etl
import models_partc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
    
    ## Creat pandas df from csv
    events, mortality, feature_map = etl.read_csv('../data/train/')
    
    
    patient_features, mortality = etl.create_features(events, mortality, feature_map)
    
    features = patient_features.copy()
    etl.save_svmlight(features, mortality, "../data/train_features_svmlight.train", "../data/train_features.train")
    
    
    events_test = pd.read_csv("../data/test/events.csv")
    map_test = pd.read_csv("../data/test/event_feature_map.csv")
    
    # Merge filtered events with event_feature_map
    agg = pd.merge(events_test, map_test, on='event_id')
    agg.dropna(inplace=True)
    
    # Group by patient and event. Then create feature value pairs.
    agg1 = agg.groupby(['patient_id', 'event_id', 'idx']).agg({'value': 'count'}).reset_index()
    feature_pairs = agg1[['patient_id','idx', 'value']]
    feature_pairs.rename(columns={'idx': 'feature_id', 'value': 'feature_value'}, inplace=True)
    
    # Normalize the value column with the formula (x-min(x))/(max(x)-min(x))
    #feature_pairs['feature_value'] = (feature_pairs['feature_value'] - feature_pairs['feature_value'].min()) / (feature_pairs['feature_value'].max() - feature_pairs['feature_value'].min())
    
    feature_pairs['max_feature_value'] = feature_pairs.groupby(['feature_id'])['feature_value'].transform(np.max)
    
    feature_pairs['feature_value'] = (feature_pairs['feature_value']) / (feature_pairs['max_feature_value'])
    
    agg_events_test = feature_pairs.copy()
    
    test_features = {}
    
    for patient in set(agg_events_test['patient_id']):
        for index, row in agg_events_test[agg_events_test['patient_id'] == patient].iterrows():
            if patient in test_features.keys():
                test_features[patient].append((row['feature_id'], row['feature_value']))
            else:
                test_features[patient] = [(row['feature_id'], row['feature_value'])]
                

    deliverable = open('../deliverables/test_features.txt', 'wb')

    for key, values in sorted(test_features.items()):
        sec_part = " ".join(str(int(val[0])) + ":" + format(val[1], '.6f') for val in sorted(values))
        deliverable.write(str(int(key)) + " " + sec_part + '\n')  

    etl.save_svmlight(test_features, mortality, '../data/test_features_svmlight.test', '../data/test_features.test')
    
    X_train, Y_train = utils.get_data_from_svmlight("../data/train_features_svmlight.train")
    X_test, Y_test = utils.get_data_from_svmlight("../data/test_features_svmlight.test")
    
    return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
    
    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)
    
    Y_pred = forest.predict(X_test)
    
    return Y_pred


def main():
    X_train, Y_train, X_test = my_features()
    Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
    utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	