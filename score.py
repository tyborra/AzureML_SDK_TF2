import joblib
import sklearn
import numpy as np
import os
import json
import pandas as pd
import tensorflow as tf
from azureml.core.model import Model
from sklearn.preprocessing import LabelEncoder
import pandas_validator as pv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class InputValidator(pv.DataFrameValidator): 
    '''
    Class to test input data validity
    '''
    row_num = 120
    column_num = 6
    userID = pv.IntegerColumnValidator('user_id', min_value=0)
    cat1 = pv.IntegerColumnValidator('cat1', min_value=0, max_value=120)
    cat2 = pv.IntegerColumnValidator('cat2', min_value=0, max_value=3)
    cat3 = pv.IntegerColumnValidator('cat3', min_value=0, max_value=40)
    numeric1 = pv.FloatColumnValidator('numeric1', min_value=0, max_value=10)
    target = pv.FloatColumnValidator('target', min_value=0, max_value=1)
    
    
class TestValidator(pv.DataFrameValidator):  
    '''
    Class to test test data validity
    '''
    row_num = 200
    column_num = 5
    userID = pv.IntegerColumnValidator('user_id', min_value=0)
    cat1 = pv.IntegerColumnValidator('cat1', min_value=0, max_value=120)
    cat2 = pv.IntegerColumnValidator('cat2', min_value=0, max_value=3)
    cat3 = pv.IntegerColumnValidator('cat3', min_value=0, max_value=40)
    numeric1 = pv.FloatColumnValidator('numeric1', min_value=0, max_value=10)
    numeric1 = pv.FloatColumnValidator('numeric1', min_value=0, max_value=10) 
    
    
def prepare_data(data_df, is_train, label_dict, scaler_dict):
    '''
    Process input data from azure
    '''
    data_df.fillna(0) 
        
    #encode features
    cat_features = ['user_id', 'cat1', 'cat2', 'cat3']
    num_features = ['numeric1']
    
    #label encoding
    data_df.replace(label_dict)
    
    #scaling
    for col in num_features:
        (data_df[col] - scaler_dict[col][0]) /  scaler_dict[col][1]
        
    #dict for tf
    df_dict= {
              'input_user_id':data_df[cat_features[0]],
              'input_cat1':data_df[cat_features[1]],
              'input_cat2':data_df[cat_features[2]],
              'input_cat3':data_df[cat_features[3]],
              'input_num_features':data_df[num_features[0]]
              } 
    
    # if true return labels
    if is_train:
        # convert target to cat
        data_df = ewma_to_cat(data_df)
        labels = data_df['target']    
        return df_dict, labels
    
    else:
        return df_dict
    

def ewma_to_cat(df):
    '''
    Convert target to discrete based on IQR
    returns: modified df
    '''    
    q1, q2, q3 = np.percentile(df['target'], [25,50,75])
    df.loc[df['target'] > q2, 'target'] = 1
    df.loc[df['target'] <= q2, 'target'] = 0
    #user_df.loc[(user_df["ewma"] >= q1) & (user_df["ewma"] <= q3), 'ewma'] = 1    #group Q2 and Q3 to make trinary
    df['target'] = df['target'].astype(int)
    return df


def cos_sim(user, user_df):
    '''
    Cosine similarity for users
    '''
    n = 10
    user_row = user_df.loc[user_df['user_id'] == user]
    user_row = user_row.drop(['user_id'], axis = 1)
    
    result_df = pd.DataFrame(user_df['user_id'])
    no_id_df = user_df.drop(['user_id'], axis = 1)

    cos_sim = cosine_similarity(no_id_df, user_row )
    result_df['cos_sim'] = cos_sim
    top_n = result_df[result_df['user_id'] != user].nlargest(10, 'cos_sim')['user_id']    
    return top_n.values


def init():
    '''
    Initialize model and globals
    '''
    global question_df
    global TFmodel      
     
    #Load TF model
    tf_model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'tf_model.h5')
    TFmodel = tf.keras.models.load_model(tf_model_path)   
    

def run(data):  #, test
    '''
    Run script
    ''' 
    
    try:
        #Input Data
        input_data = pd.DataFrame(json.loads(data)['data'])
        test_data =  pd.DataFrame(json.loads(data)['test'])  
        rec_data =   pd.DataFrame(json.loads(data)['rec']) 
        label_dict = json.loads(data, object_hook=lambda d: 
                                {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})['label_dict']
        scaler_dict = json.loads(data)['scaler_dict']
        
        #Top ten similar users
        user = input_data['user_id'][0]
        top_ten = cos_sim(user, rec_data)
        
    except AssertionError as error:
         return error 

    #Validate Data
    input_validator = InputValidator()
    test_validator = TestValidator()
    
    #user data
    try:
        assert(input_validator.is_valid(input_data))
    except:
        return ('Assertion error, invalid User data')
    
    #question data
    try:
        assert(test_validator.is_valid(test_data))
    except:
        return ('Assertion error, invalid Question data')

    #process input and convert dict and labels
    input_dict, labels = prepare_data(input_data, True, label_dict, scaler_dict)  
    test_dict = prepare_data(test_data, False, label_dict, scaler_dict)

    #append TF model with input data
    TFmodel.fit(input_dict, labels, epochs= 5)    
    result = TFmodel.predict(test_dict)
    
    return result.tolist(), top_ten.tolist()  
