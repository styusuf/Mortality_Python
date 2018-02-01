import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    
    morts = mortality['patient_id']
    dead = events[events['patient_id'].isin(morts)]
    alive = events[~events['patient_id'].isin(morts)]
    
    alive_group = alive.groupby(['patient_id']).size().reset_index(name='count')
    dead_group = dead.groupby(['patient_id']).size().reset_index(name='count')
    
    
    avg_dead_event_count = dead_group['count'].mean()
    max_dead_event_count = dead_group['count'].max()
    min_dead_event_count = dead_group['count'].min()
    avg_alive_event_count = alive_group['count'].mean()
    max_alive_event_count = alive_group['count'].max()
    min_alive_event_count = alive_group['count'].min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    
    morts = mortality['patient_id']
    dead = events[events['patient_id'].isin(morts)]
    alive = events[~events['patient_id'].isin(morts)]
    
    alive_group = alive.groupby(['patient_id', 'timestamp']).size().reset_index(name='count')
    dead_group = dead.groupby(['patient_id', 'timestamp']).size().reset_index(name='count')

    alive_encounter_count = alive_group.groupby(['patient_id']).agg('count').reset_index()
    alive_encounter_count.drop('timestamp', axis=1, inplace=True)
    dead_encounter_count = dead_group.groupby(['patient_id']).agg('count').reset_index()
    dead_encounter_count.drop('timestamp', axis=1, inplace=True)
    
    avg_dead_encounter_count = round(dead_encounter_count['count'].mean(),1)
    max_dead_encounter_count = dead_encounter_count['count'].max()
    min_dead_encounter_count = dead_encounter_count['count'].min()
    avg_alive_encounter_count = round(alive_encounter_count['count'].mean(), 1)
    max_alive_encounter_count = alive_encounter_count['count'].max()
    min_alive_encounter_count = alive_encounter_count['count'].min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    
    morts = mortality['patient_id']
    dead = events[events['patient_id'].isin(morts)]
    alive = events[~events['patient_id'].isin(morts)]
    
    alive['timestamp'] = pd.to_datetime(alive['timestamp'])
    a_g1 = alive.groupby(['patient_id'])
    a_g2 = a_g1.agg(lambda x: x['timestamp'].max() - x['timestamp'].min()).reset_index()
    a_g2['value'] = a_g2['value'].dt.days
    alive_record_length = a_g2[['patient_id', 'value']]
    
    dead['timestamp'] = pd.to_datetime(dead['timestamp'])
    m_g1 = dead.groupby(['patient_id'])
    m_g2 = m_g1.agg(lambda x: x['timestamp'].max() - x['timestamp'].min()).reset_index()
    m_g2['value'] = m_g2['value'].dt.days
    dead_record_length = m_g2[['patient_id', 'value']]
    
    avg_dead_rec_len = round(dead_record_length['value'].mean(),1)
    max_dead_rec_len = dead_record_length['value'].max()
    min_dead_rec_len = dead_record_length['value'].min()
    avg_alive_rec_len = round(alive_record_length['value'].mean(),1)
    max_alive_rec_len = alive_record_length['value'].max()
    min_alive_rec_len = alive_record_length['value'].min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'
    #train_path = '../tests/data/statistics/'
    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
