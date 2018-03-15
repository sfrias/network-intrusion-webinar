import urllib
import gzip
import os
import sys 
import pickle
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier
from IPython.display import clear_output, Image

warnings.filterwarnings('ignore')


def scale_data(array):
    if len(array.shape) == 1:
        return StandardScaler().fit_transform(array[:, np.newaxis])
    else:
        return StandardScaler().fit_transform(array)

def download(sample=True):

    attacktypespath = 'attacktypes.pkl'
    if sample:
        df_reader = pd
        targetpath = '../kddcup.data_10_percent'
        URL = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
    else:
        df_reader = dd
        targetpath = '../kddcup.data'
        URL = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
    
    compressedpath = '{}.gz'.format(targetpath)        
    ATTACK_TYPES_URL = 'http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types'
    
    if not os.path.exists(targetpath):
        
        if os.path.exists(compressedpath):
            os.remove(compressedpath)
            
        contents = urllib.request.urlopen(URL).read()    
        
        with open(compressedpath,'wb') as f:
            f.write(contents)

        with gzip.GzipFile(compressedpath, 'rb') as gz_file:
            contents = gz_file.read()

        with open(targetpath,'wb') as outfile:
            outfile.write(contents)
            
    
    else:
        pass
        

    attack_info = urllib.request.urlopen(ATTACK_TYPES_URL).readlines()
    attack_types = {'normal':'normal'}
        
    for attack, attack_type in filter(lambda x: len(x) > 0, map(bytes.split,attack_info)):
        attack_types[attack.decode()] = attack_type.decode()
            
    return targetpath, attack_types, df_reader

def load_dataframe(path, df_reader):
    
    df = df_reader.read_csv(path, header=None)

    features = [
        'duration'
        ,'protocol_type'
        ,'service'
        ,'flag'
        ,'src_bytes'
        ,'dst_bytes'
        ,'land'
        ,'wrong_fragment'
        ,'urgent'
        ,'hot'
        ,'num_failed_logins'
        ,'logged_in'
        ,'num_compromised'
        ,'root_shell'
        ,'su_attempted'
        ,'num_root'
        ,'num_file_creations'
        ,'num_shells'
        ,'num_access_files'
        ,'num_outbound_cmds'
        ,'is_host_login'
        ,'is_guest_login'
        ,'count'
        ,'srv_count'
        ,'serror_rate'
        ,'srv_serror_rate'
        ,'rerror_rate'
        ,'srv_rerror_rate'
        ,'same_srv_rate'
        ,'diff_srv_rate'
        ,'srv_diff_host_rate'
        ,'dst_host_count'
        ,'dst_host_srv_count'
        ,'dst_host_same_srv_rate'
        ,'dst_host_diff_srv_rate'
        ,'dst_host_same_src_port_rate' 
        ,'dst_host_srv_diff_host_rate'
        ,'dst_host_serror_rate'
        ,'dst_host_srv_serror_rate'
        ,'dst_host_rerror_rate'
        ,'dst_host_srv_rerror_rate'
        ,'class'
    ] 

    feature_columns = df.columns.values.tolist()
    
    for i, name in enumerate(features):
        feature_columns[i] = name
    
    df.columns = feature_columns
    
    df['class'] = df['class'].apply(lambda x: x.replace('.',''))
    df = df.drop_duplicates()
    return df


def visualize_tree(estimator, X, y, feature_names=None, no_color='red', yes_color='green', title="", graph_size="9.75,18.25"):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    features = estimator.tree_.feature
    thresholds = estimator.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    graph = Digraph(graph_attr={'size':graph_size})
    
    if feature_names is None:
        feature_names = ["Feature {}".format(i) for i in range(X.shape[1])]
    
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    
    rules = {}
    for i in range(n_nodes):
        if is_leaves[i]:
            continue
        else:
            left, right, threshold, feature = children_left[i], children_right[i], thresholds[i], features[i]        

            filter_right = X[:, feature] > threshold 
            filter_left = X[:, feature] < threshold 

            rules[left] = (i, filter_left)
            rules[right] = (i, filter_right)

            feature = feature_names[feature]

            graph.node(str(i), label="{0}\n<={1}".format(feature, round(threshold, 2)))
            graph.node(str(right))
            graph.node(str(left))
            graph.edge(str(i), str(right), label="no", color=no_color)
            graph.edge(str(i), str(left), label="yes", color=yes_color)


    f_rules = {}
    def getit(key, current_filter, rule_dict):
        if key not in rule_dict:
            return current_filter
        else:
            parent, next_filter = rule_dict[key]
            return current_filter & getit(parent, next_filter, rule_dict)

    for i in range(n_nodes):
        if is_leaves[i]:
            default = np.ones(X.shape[0]).astype(bool)
            final_filter = getit(i, default, rules)
            f_rules[i] = final_filter

    attack_emojis ={
        'normal':"😃",
        'u2r':"☠️",
        'dos':"🤖",
        'probe':"👀",
        'r2l':"🕵",}    
    
    for leaf_node in f_rules:
        data = X[f_rules[leaf_node]]

        vals = pd.Series(y[f_rules[leaf_node]]).value_counts(normalize=True)

        label = ""
        for attack_type, prop in vals.iteritems():

            label += "{0}: {1}{2}\n".format(attack_type, round(prop, 2), attack_emojis[attack_type])

        graph.node(str(leaf_node), label=label)

    styles = {
        'graph': {
            'label': title,
            'fontsize': '16',
            'fontcolor': 'white',
            'bgcolor': '#333333',
            'rankdir': 'TB',
        },
        'nodes': {
            'fontname': 'Helvetica',
            'fontcolor': 'white',
            'color': 'white',
            'style': 'filled',
            'fillcolor': '#006699',
            'width':"0.3"
        },
        'edges': {
            'style': 'dashed',
            'color': 'white',
            'arrowhead': 'open',
            'fontname': 'Courier',
            'fontsize': '12',
            'fontcolor': 'white',

        }
    }

    def apply_styles(graph, styles):
        graph.graph_attr.update(
            ('graph' in styles and styles['graph']) or {}
        )
        graph.node_attr.update(
            ('nodes' in styles and styles['nodes']) or {}
        )
        graph.edge_attr.update(
            ('edges' in styles and styles['edges']) or {}
        )
        return graph


    graph = apply_styles(graph, styles)
    return graph