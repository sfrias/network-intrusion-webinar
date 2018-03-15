# Imports
import yaml
import urllib
import gzip
import pickle
import os
import pandas as pd
import numpy as np
import sys
import boto3
from botocore.exceptions import ClientError
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


CWD = os.getcwd()
HOME = os.path.expanduser("~")

class Utilities(object):
    
    def scale_data(array):
        if len(array.shape) == 1:
            return StandardScaler().fit_transform(array[:, np.newaxis])
        else:
            return StandardScaler().fit_transform(array)
    def safe_ratio(x, y):
        if x + y > 0:
            return x / (x + y)
        else:
            return -1 
        
    def invert_dict(d):
        return {value: key for key, value in d.items()}  
    
        
def retrieve_attack_types(url):
    print("Downloading attack type information")    
    attack_info = urllib.request.urlopen(url).readlines()
    attack_types = {'normal':'normal'}
    non_empty_attacktypes = filter(lambda x: len(x) > 0, map(bytes.split, attack_info))
    
    for attack, attack_type in non_empty_attacktypes:
        attack_types[attack.decode()] = attack_type.decode()  
        
    print("Attack type information downloaded!")            
    return attack_types

def download_data_from_source(data_url, data_path):
    
    if os.path.exists(data_path):
        os.remove(target_path)
        
    compressed_path = "{}.gz".format(data_path)
    
    if os.path.exists(compressed_path):
        os.remove(compressed_path)
            
    contents = urllib.request.urlopen(data_url).read()    
        
    with open(compressed_path,'wb') as f:
        f.write(contents)

    with gzip.GzipFile(compressed_path, 'rb') as gz_file:
        contents = gz_file.read()

    with open(data_path,'wb') as outfile:
        outfile.write(contents)
        
    print("Data downloaded and saved to {}".format(data_path))
        
    return data_path

def load_dataframe(path, columns):
    import dask.dataframe as dd
    df = dd.read_csv(path, header=None)
    df.columns = columns
    print("dataframe loaded with columns: {}".format(columns))
    return df

def load_model(config):
    
    model_path = os.path.join(CWD, config['model_filename'])
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
        
    return model

def preprocess_and_sample_dataframe(df, attack_types):
    
    df = df.drop_duplicates()
    df['class'] = df['class'].apply(lambda x: x.replace('.',''))
    df['attack_type'] = df['class'].map(attack_types)
    return df.sample(0.1).compute()

def engineer_features(df, feature_list):
    
    feature_transforms = {
        'is_flag_S0': [['flag'], lambda x: x['flag'] == 'S0'],
        'is_flag_REJ': [['flag'], lambda x: x['flag'] == 'REJ'],
        'is_flag_RSTR': [['flag'], lambda x: x['flag'] == 'RSTR'],
        'is_service_FTP': [['service'], lambda x: x['service'] in ('ftp', 'ftp_data')],
        'is_service_private': [['service'], lambda x: x['service'] == 'private'],
        'is_service_eco_i': [['service'], lambda x: x['service'] == 'eco_i'],
        'is_service_other': [['service'], lambda x: x['service'] == 'other'],
        'src_dst_ratio': [['src_bytes','dst_bytes'], lambda x: Utilities.safe_ratio(x['src_bytes'], x['dst_bytes'])]
    }
    
    for new_feature in feature_transforms:
        based_on, func = feature_transforms[new_feature]
        df[new_feature] = df[based_on].apply(func, axis=1)
        
        feature_list.append(new_feature)
        print("{} engineered!".format(new_feature))
       
    return df

def prepare_training_data(df, features, class_map, class_column='attack_type', scaled=True):

    X = df[features].values.astype(float)
    
    if scaled:
        X = Utilities.scale_data(X)
    
    y = df['attack_type'].map(class_map).values
    
    return X, y, class_map


def fit_model(X, y):
    print("Fitting model...")
    model = GradientBoostingClassifier()
    model.fit(X, y)
    print("Model fit!")
    return model

def download_model_from_s3(config):
    model_path = os.path.join(CWD, config['model_filename'])
    s3_key = config['model_s3_key']
    download_from_s3(s3_key, model_path)

def save_model(model, config):
    model_path = os.path.join(CWD, config['model_filename'])
    
    if os.path.exists(model_path):
        os.remove(model_path)
        
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
        
    s3_key = config['model_s3_key']
    upload_to_s3(model_path, s3_key)
        
    print("Model saved to {}".format(model_path))
    
    
    
def save_full_dataset():
    """Full dataset is a """
    FULL_DATA_PATH = os.path.join(HOME, config['full_data_filename'])    
    AWS_KEY = config['full_data_s3_key']
    upload_to_s3(FULL_DATA_PATH, AWS_KEY)
    print("Full dataset uploaded to s3 as {}".format(AWS_KEY))
    
    
def save_quick_dataset(sample):
    """Quick dataset is a Pandas dataframe"""
    QUICK_DATA_PATH = os.path.join(CWD, config['quick_data_filename'])    
    sample.to_csv(QUICK_DATA_PATH)
    print("Sample dataset saved to {}".format(QUICK_DATA_PATH))
    AWS_KEY = config['quick_data_s3_key']
    upload_to_s3(QUICK_DATA_PATH, AWS_KEY)    
    print("Full dataset uploaded to s3 as {}".format(AWS_KEY))
    
        
        
def train(config):
    data_url = config['data_url']
    attack_types_url = config['attack_types_url']
    data_path = os.path.join(HOME, config['full_data_filename'])
    model_path = os.path.join(CWD, config['model_filename'])
    traffic_features = config['traffic_columns']
    content_features = config['content_columns']
    basic_features = config['basic_columns']
    columns = config['df_columns']
    non_engineered_features = traffic_features + content_features + basic_features
    all_features = config['features']
    data_s3_key = config['full_data_s3_key']
    
    classmap = {}
    for entry in config['classmap']:
        classmap.update(entry)
        
    int_to_class = config['classmap']
    
    try:
        print("Trying to load training from local file.")
        sample = load_sample_data(config)
    except FileNotFoundError:
        print("...Failed, trying to load from s3")
        download_sample_data_from_s3(config)
        sample = load_sample_data(config)
    except:            
        print("...Failed, rebuilding from full dataset...")
        data_path = get_full_data(config)
        attack_types = retrieve_attack_types(attack_types_url)
        df = load_dataframe(data_path, columns)
        sample = preprocess_and_sample_dataframe(df, attack_types)
        sample = engineer_features(sample, non_engineered_features)
    save_sample_data(sample, config)
    X, y, class_map = prepare_training_data(sample, all_features, classmap)
    model = fit_model(X, y)
    save_model(model, config)

    return model, int_to_class

def load_config(path='config.yml'):
    with open(path) as config_file:
        config = yaml.load(config_file.read())
    return config


def download_from_s3(Key, Filename, bucket='ds-cloud-public-shared'):
    key = os.environ['AWS_KEY']
    secret = os.environ['AWS_SECRET']
    s3 = boto3.resource('s3', aws_access_key_id = key, aws_secret_access_key = secret)
    
    bucket = s3.Bucket(bucket)
    bucket.download_file(Key, Filename) 
    
    return Filename

def upload_to_s3(Filename, Key, bucket='ds-cloud-public-shared'):
    key = os.environ['AWS_KEY']
    secret = os.environ['AWS_SECRET']    
    s3 = boto3.resource('s3', aws_access_key_id = key, aws_secret_access_key = secret)
    
    bucket = s3.Bucket(bucket)
    bucket.upload_file(Filename, Key)   
    
def save_sample_data(sample, config):
    
    path = os.path.join(CWD, config['quick_data_filename'])
    s3_key = config['quick_data_s3_key']
    sample.to_csv(path)
    upload_to_s3(path, s3_key)
    print("Sample saved to {0} and uploaded to {1} on s3".format(path, s3_key))

def load_sample_data(config):
    path = os.path.join(CWD, config['quick_data_filename'])
    return pd.read_csv(path)


def download_sample_data_from_s3(config):
    path = os.path.join(CWD, config['quick_data_filename'])
    s3_key = config['quick_data_s3_key']

    download_from_s3(s3_key, path)
        
def get_full_data(config):
    
    data_url = config['data_url']
    data_path = os.path.join(HOME, config['full_data_filename'])
    data_s3_key = config['full_data_s3_key']
    
    if not os.path.exists(data_path):
        print("Data not found locally, trying s3...")
        try:
            data_path = download_from_s3(data_s3_key, data_path)
            print("Data downloaded from s3")
        except (FileNotFoundError, ClientError) as not_found:
            print("Data not on s3, rebuilding")
            data_path = download_data_from_source(data_url, data_path)
            upload_to_s3(data_path, data_s3_key)
            print("data downloaded from source and uploaded to s3")
    else:
        print("Data already exists, proceeding without download")
    
    return data_path

def log_state():
    cwd = os.getcwd()
    files = []
    for (dirpath, dirnames, filenames) in os.walk(cwd):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    
    print("Working Directory: {}".format(cwd))
    print("Files: {}".format(files))
