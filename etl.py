import utils
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    
    morts = mortality['patient_id'] #patient_id of dead patients
    alive = events[~events.patient_id.isin(morts)] #separate living patients
    
    # Compute index date of dead patients
    mortality['timestamp'] = pd.to_datetime(mortality['timestamp']).dt.date
    mortality['indx_date'] = mortality['timestamp'] - pd.to_timedelta(30, unit='d')
    mortality.head()
    dead = mortality[['patient_id', 'indx_date']]
    
    # Compute index date of living patients
    a_g1 = alive.groupby(['patient_id']).agg({'timestamp': {'indx_date': 'max'}}).reset_index()
    a_g1.columns = ['patient_id', 'indx_date']
    
    #combine both dataset
    indx_date = dead.append(a_g1)

    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    
    evts = pd.merge(events, indx_date, on='patient_id')
    evts['timestamp'] = pd.to_datetime(evts['timestamp']).dt.date
    evts['indx_date'] = pd.to_datetime(evts['indx_date']).dt.date
    filtered_events = evts[(evts['timestamp'] <= evts['indx_date']) & (evts['timestamp'] >= evts['indx_date'] - pd.to_timedelta(2000, unit='d'))]
    
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    # Merge filtered events with event_feature_map
    agg = pd.merge(filtered_events_df, feature_map_df, on='event_id')
    agg.dropna(inplace=True)
    
    # Group by patient and event. Then create feature value pairs.
    agg1 = agg.groupby(['patient_id', 'event_id', 'idx']).agg({'value': 'count'}).reset_index()
    feature_pairs = agg1[['patient_id','idx', 'value']]
    feature_pairs.rename(columns={'idx': 'feature_id', 'value': 'feature_value'}, inplace=True)
    
    # Normalize the value column with the formula (x-min(x))/(max(x)-min(x))
    #feature_pairs['feature_value'] = (feature_pairs['feature_value'] - feature_pairs['feature_value'].min()) / (feature_pairs['feature_value'].max() - feature_pairs['feature_value'].min())
    
    feature_pairs['max_feature_value'] = feature_pairs.groupby(['feature_id'])['feature_value'].transform(np.max)
    
    feature_pairs['feature_value'] = (feature_pairs['feature_value']) / (feature_pairs['max_feature_value'])
    
    
    aggregated_events = feature_pairs.copy()
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    patient_features = {}
    mortality_dict = {}
    
    for patient in set(aggregated_events['patient_id']):
        for index, row in aggregated_events[aggregated_events['patient_id'] == patient].iterrows():
            if patient in patient_features.keys():
                patient_features[patient].append((row['feature_id'], row['feature_value']))
            else:
                patient_features[patient] = [(row['feature_id'], row['feature_value'])]
    
    for index, row in mortality.iterrows():
        mortality_dict[row['patient_id']] = row['label']

    return patient_features, mortality_dict

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    features = patient_features.copy()
    for key in patient_features.keys():
        if key in mortality.keys():
            features[key].insert(0,1)
        else:
            features[key].insert(0,0)
            
    for key, values in sorted(features.items()):
        first_part = str(values[0]) + " "
        sec_part = " ".join(str(int(val[0])) + ":" + format(val[1], '.6f') for val in sorted(features[key][1:]))
        deliverable1.write(first_part + sec_part + '\n')
        deliverable2.write(str(int(key)) + " " + first_part + sec_part + " " + '\n')

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()