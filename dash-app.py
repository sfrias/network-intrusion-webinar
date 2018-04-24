import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import os
import sys
import pandas as pd
import numpy as np
from textwrap import dedent as s
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
#from graphviz import Digraph
import skater


sys.path.insert(0,'utilities/')
from plotly_webinar_utils import download, load_dataframe, visualize_tree
from core import load_config

####################
#
# Styling / theme
#
####################
textColor = 'lightgrey'

sectionStyle = {
    'padding': 20,
    'margin': 20,
    'borderRadius': 5,
    'border': 'solid',
    'borderColor': 'lightgrey'
}

sectionStyleNB = {
    'padding': 20,
    'margin': 20,
    'borderRadius': 5,
    'borderColor': 'lightgrey'
}

traceColors = [
    "rgb(211,255,255)",
    "rgb(159,216,255)",
    "rgb(119,181,255)",
    "rgb(76,148,253)",
    "#0074D9"
    ]

####################
#
# Read in data
#
####################
path, attack_types, df_reader = download(sample=True)
df = load_dataframe(path, df_reader)
df['attack_type'] = df['class'].map(attack_types)
df1 = df
config = load_config()

####################
#
# Exploration of the Dataset with Plotly
#
####################
def event_data_table(df):
    """
    """
    nb_examples=1000
    df = df[0:nb_examples]
    color_vals = ['rgb(30, 30, 30)'] * nb_examples

    # high data entries :
    ids = df.loc[df['class'] != 'normal'].index.values
    for i in ids:
        color_vals[i] = 'red'
    trace = go.Table(header=dict(values=[
        '<b>Duration','<b>Protocol Type','<b>Service',
        '<b>Flag', '<b>src_bytes', '<b>dst_bytes', '<b>class', '<b>attack_type'],
         fill=dict(color='rgb(30, 30, 30)'),
         line=dict(color='rgb(30, 30, 30)'),
         font=dict(size=14),
         align=['left'] * 5),

         cells=dict(values=[df.duration, df.protocol_type,
                            df.service, df.flag,
                            df.src_bytes, df.dst_bytes, df['class'],
                            df['attack_type']],
                    fill=dict(color=[color_vals]),
                    line=dict(color='rgb(30, 30, 30)'),
                    align=['left']*5))
    data = [trace]
    layout = go.Layout(
        plot_bgcolor = 'rgb(30, 30, 30)',
        paper_bgcolor = 'rgb(30, 30, 30)',
        font = dict(
            color = textColor
        ),
        margin = dict(
            t = 20, b = 20, l = 20, r = 20
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

####################
#
# Frequency Count of Connection entries by Class (Normal vs Attack)
#
####################
def chart_attack_counts(df1, chart_type='bar'):
    """ Provides normalized event counts per event type
    """
    counts = df1['attack_type'].value_counts(normalize=True).sort_values()
    counts = counts*100

    if chart_type == 'bar':
        count_data = [go.Bar(
            x=list(counts.index),
            y=list(counts),
            marker=dict(color=traceColors)
        )]
    elif chart_type == 'pie':
        count_data = [go.Pie(labels=counts.index, values=counts.values,
                            marker=dict(colors=traceColors))]
    else:
        raise ValueError("Wrong chart type. Options are 'bar' or 'pie'")

    layout = go.Layout(
            title='Frequency Distribution of Events by Type',
            xaxis=dict(title='Attack Type', showgrid=True, gridcolor=textColor),
            yaxis=dict(title='Percentage (%)', showgrid=True, gridcolor=textColor),
            plot_bgcolor = 'rgb(30, 30, 30)',
            paper_bgcolor = 'rgb(30, 30, 30)',
            font = dict(
                color = textColor
            ),
            margin = dict(
                t = 60, b = 60, l = 60, r = 20
            )
        )

    fig = go.Figure(data=count_data, layout=layout)

    return fig

# generate html table
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

####################
#
# Splitting the features among the "basic", "content" and "traffic" features
#
####################
d1 = df1[config['content_columns']].dtypes
d2 = df1[config['basic_columns']].dtypes
d3 = df1[config['traffic_columns']].dtypes
max_rows = max(map(len, [d1,d2,d3]))

cols = {}
cols1 = []

for feature, type_ in d1.iteritems():
    cols1.append(": ".join([feature, str(type_)]))

cols1 = cols1 + [""] * (max_rows - len(cols1))

cols['Content Attributes']=cols1
cols2 = []
for feature, type_ in d2.iteritems():
    cols2.append(": ".join([feature, str(type_)]))

cols2 = cols2 + [""] * (max_rows - len(cols2))

cols['Basic TCP Attributes']=cols2
cols3 = []
for feature, type_ in d3.iteritems():
    cols3.append(": ".join([feature, str(type_)]))

cols3 = cols3 + [""] * (max_rows - len(cols3))
cols['Traffic Attributes']=cols3
cols2 = pd.DataFrame.from_dict(cols)

####################
#
# Plotly Heatmap : Response vs Traffic Predictors
#
####################
def mycut(series, bins, labels=None):
    """ COMPLETE THIS

    """
    groups = {}
    names = {}
    percentiles = pd.Series(np.percentile(series, [i * 5 for i in range(21)]))
    unique_percentile_vals = np.unique(percentiles)
    data = series.apply(lambda x: unique_percentile_vals[
        np.argmin(abs(unique_percentile_vals-x))])
    for val in unique_percentile_vals:
        index = percentiles[percentiles==val].index.values
        m0, m1 = min(index) * 5, max(index) * 5
        m_ = (m0 + m1) / 2
        groups[val] = m_
        names[m_] = '{0} to {1} percentile'.format(m0, m1)
    data = data.map(groups)
    return data, names

traffic_medians = {}

columns_of_interest = config['traffic_columns']

for i, feature in enumerate(columns_of_interest):
    scaled_data, names = mycut(df1[feature], 20)
    df1['aux'] = scaled_data
    medians = df1.groupby('attack_type')['aux'].median()
    traffic_medians[feature] = medians

df1 = df1.drop('aux', 1)

medians = pd.DataFrame(traffic_medians).T.unstack().reset_index()
medians.columns = ['attack_type', 'network attribute', 'typical percentile']


# Utility Functions:
def scale_data(array):
    """Standardize features by removing the mean and scaling
    to unit variance
    """
    if len(array.shape) == 1:
        return StandardScaler().fit_transform(array[:, np.newaxis])
    else:
        return StandardScaler().fit_transform(array)


def safe_ratio(src_bytes, dst_bytes):
    """ Computes fractional contribution of data size sent
    from source vs destination.
    """

    if src_bytes + dst_bytes > 0:
        return src_bytes / (src_bytes + dst_bytes)
    else:
        return -1

####################
#
# Feature Engineering
#
####################
feature_transforms = {
    'is_flag_S0': [['flag'], lambda x: x['flag'] == 'S0'],
    'is_flag_REJ': [['flag'], lambda x: x['flag'] == 'REJ'],
    'is_flag_RSTR': [['flag'], lambda x: x['flag'] == 'RSTR'],
    'is_service_FTP': [['service'], lambda x: x['service'] in ('ftp', 'ftp_data')],
    'is_service_private': [['service'], lambda x: x['service'] == 'private'],
    'is_service_eco_i': [['service'], lambda x: x['service'] == 'eco_i'],
    'is_service_other': [['service'], lambda x: x['service'] == 'other'],
    'src_dst_ratio': [['src_bytes','dst_bytes'], lambda x: safe_ratio(x['src_bytes'],
                                                                      x['dst_bytes'])]
}

# Let's apply these transformations on the relevant columns:
for new_feature in feature_transforms:
    based_on, func = feature_transforms[new_feature]
    df1[new_feature] = df1[based_on].apply(func, axis=1)

basic_columns = config['basic_columns']
content_columns = config['content_columns']
traffic_columns = config['traffic_columns']

# Newly engineered features:
features = [
    'is_service_eco_i'
    ,'is_flag_REJ'
    ,'is_service_private'
    ,'is_service_FTP'
    ,'src_dst_ratio'
    ,'is_flag_RSTR'
    ,'is_flag_S0'
    ,'is_service_other'
]

features.extend(traffic_columns)
features.extend(content_columns)
features.extend(basic_columns)

####################
#
# Train
#
####################
from sklearn.model_selection import train_test_split

# Map each class of attack into an integer.
# This could be useful for future reference.
unique_classes = df1['attack_type'].unique()
class_map = {j: i for i, j in enumerate(unique_classes)}

# Convert predictors and target variables to numpy arrays:
X = df1[features].values.astype(float)
X_scaled = scale_data(X)
y = df1['attack_type'].values

# Split the dataset into train/test sets. We set the test set to be 1/3
# of the whole dataset:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33)

# Here we increase the weights of the minority (attack) classes. Minority
# classes are more heavily weighted to force the classifier to identify
# these rare events.
class_weight_vals = (1 - (pd.Series(y).value_counts(normalize=True))).to_dict()
sample_weights_train = np.array([class_weight_vals[i] for i in y_train])
sample_weights_test = np.array([class_weight_vals[i] for i in y_test])


####################
#
# dash app / layout / callbacks
#
####################
app = dash.Dash()

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions']=True

app.layout = html.Div([

    html.Div([
        html.Div([
            html.H2('NETWORK INTRUSION IN THE AGE OF IOT')
        ], className = "eight columns"),

        html.Div([
            html.Img(
                src=('https://s3-us-west-1.amazonaws.com/plotly-tutorials/'
                'logo/new-branding/dash-logo-by-plotly-stripe.png'),
                style={
                    'height': '100px',
                    'float': 'right'}),
        ], className = "four columns")
    ], className = "row"),

    html.Hr([]),

    html.Div(
        dcc.Tabs(
            tabs=[
                {'label': 'Exploratory Data Analysis', 'value': 1},
                {'label': 'Classification Model', 'value': 2},
                {'label': 'References', 'value': 3}
            ],
            value=1,
            id='tabs',
            vertical=False,
            style={
                'color': textColor,
                'backgroundColor': 'rgb(30, 30, 30)'
            }
        ),
        style={'width': '100%', 'float': 'left'}
    ),

    html.Div([
        #tab 1
        html.Div([
            html.Section([
                html.Div(id='tab-output')
            ], style={
                'padding': 20,
                'margin': 0,
                'borderRadius': 5,
                'border': 'solid',
                'borderColor': textColor
            })
        ])
    ])
])

@app.callback(Output('tab-output', 'children'), [Input('tabs', 'value')])
def display_content(value):
    if value == 1:
        tab_layout = html.Div([
            html.Br([]),
            html.Br([]),
            html.Br([]),
            html.Div([
                html.Section(
                    [html.Strong('Introduction'),
                    html.Hr([]),
                    dcc.Markdown(s(
                    '''
                    With so many internet connected devices ranging from
                    toasters, dishwashers, fridges to cars, network security of
                    these devices is paramount. It is crucial for manufacturers
                    to understand the signature and identify attacks before
                    they happen.

                    This Dash application compliments the jupyter notebooks
                    found in the repo. Here, we will use Plotly to create
                    interactive charts and tables to explore the KDD Cup 99
                    dataset. This canonical dataset has been extensively used
                    to train anomaly detection algorithms. It comes from about
                    4 Gb of compressed data of network traffic (7 weeks).
                    It contains 5M connection records, each record comprises
                    41 features. The training dataset of 4.9M entries has each
                    record labeled as either a normal connection or an attack.

                    If the connection falls under the attack label, the dataset
                    also provides the category of the attack. There are four
                    categories:

                    - Denial of Service (DoS) Attack : Overload the resource
                    with requests. Denying other legitimate users of the
                    service. We have seen this example not too long ago.

                    - User to Root Attack (U2R) : attacker gets access to a
                    user account on the host, then gets root access

                    - Remote to Local Attack (R2L) : attacker has ability to
                    send packets to a machine over the network but does not
                    have an account on that machine. User exploits vulnerability
                    to gain user access of that machine.

                    - Probing Attack : attempt to gather info about a network
                    of computers to exploit its security vulnerability.
                    '''))],
                    style=sectionStyleNB
                )
            ], className="row"),

            html.Div([
                html.Section(
                    [html.Strong('Exploration of the Dataset'),
                    html.Hr([]),
                    dcc.Graph(id='table', figure=event_data_table(df1))],
                    style=sectionStyle
                )
            ], className="row"),

            html.Div([

                html.Div([
                    html.Section(
                        [html.Strong('Frequency Count of Connection entries by \
                        Class (Normal vs Attack)'),
                        html.Hr([]),
                        dcc.Markdown(s(
                        '''
                        The bar chart below shows that almost 60.3% of all
                        entries in the dataset correspond to normal connections
                        . Denial-of-Service (DoS) is the most frequent class
                        of attacks with almost 40% of all connections.''')),
                        html.Br([]),
                        dcc.Markdown(s(
                        '''
                        **These 4 clases are attack are defined as:**

                        - DOS: denial-of-service, e.g. syn flood;
                        - R2L: unauthorized access from a remote machine, e.g.
                        guessing password;
                        - U2R: unauthorized access to local superuser (root)
                        privileges, e.g., various buffer overflow attacks;
                        - probing: surveillance and other probing, e.g., port
                        scanning.
                        '''
                        ))
                        ], style=sectionStyleNB
                    )
                ], className="six columns"),

                html.Div([
                    html.Section(
                        [dcc.Dropdown(
                            id='frequency-dropdown',
                            options=[{'label': 'Bar', 'value': 'Bar'},
                                    {'label': 'Pie', 'value': 'Pie'}],
                            value='Bar'
                        ),
                        dcc.Graph(id='frequency-graph')],
                        style=sectionStyle
                    )
                ], className="six columns"),

            ], className="row"),

            html.Div([

                html.Div([
                    html.Section(
                        [html.Strong('Categories of Predictors/Features Table'),
                        html.Hr([]),
                        generate_table(cols2)],
                        style=sectionStyle
                    )
                ], className="six columns"),

                html.Div([
                    html.Section(
                        [html.Strong('Categories of Predictors/Features'),
                        html.Hr([]),
                        dcc.Markdown(s(
                        '''
                        A description of each feature can be found on the UCI
                        KDD 99 page. In brief, the predictors are split in
                        three categories : *basic*, *content*, and *traffic
                        features*.


                        - basic features : About individual TCP connections.
                        Duration, number of bytes transfered, etc.
                        - content features : Derived features based on domain
                        knowledge and expertise of network intrusion. These
                        features look for suspicious behavior, such as the
                        number of failed login attempt (see KDD 99 page)
                        - traffic features : these are features computed within
                        a two-second time window. These include things like the
                        fraction of connections to the same service, etc.
                        '''
                        ))
                        ], style=sectionStyleNB
                    )
                ], className="six columns")

            ], className="row"),

            html.Div([
                html.Section(
                    [dcc.Graph(id='heatmap', figure={
                        'data': [
                            go.Heatmap(
                                z=medians['typical percentile'],
                                y=medians['attack_type'],
                                x=medians['network attribute'],
                                colorscale = 'Blues',
                                xaxis='hello'
                            )],
                        'layout': go.Layout(
                            title='Typical Percentile of Traffic Volume by \
                            Attack Type',
                            xaxis={'title':"Connection Attributes"},
                            yaxis={'title':"Activity Type"},
                            plot_bgcolor = 'rgb(30, 30, 30)',
                            paper_bgcolor = 'rgb(30, 30, 30)',
                            font = dict(
                                color = textColor
                            ),
                            margin = dict(
                                t = 60, b = 60, l = 60, r = 20
                            ))
                    })],
                    style=sectionStyle
                )
            ], className="row")
        ])
    elif value == 2:
        tab_layout = html.Div([
            html.Br([]),
            html.Br([]),
            html.Br([]),
            html.Div([
                html.Section(
                    [html.Button(id='run-button', n_clicks=0, children='Run Model')],
                    style={
                        'padding': 20,
                        'margin': 20,
                        'borderRadius': 5,
                        'borderColor': 'lightgrey',
                        'text-align': 'center'
                    }
                )
            ], className="row"),

            html.Div([
                html.Div([
                    html.Section(
                        [html.Strong('Model Performance'),
                        html.Hr([]),
                        dcc.Graph(id="confusion-matrix")],
                        style=sectionStyle
                    )
                ], className="six columns"),

                html.Div([
                    html.Section(
                        [html.Strong('Feature/Predictor Importance'),
                        html.Hr([]),
                        dcc.Graph(id="importance-bar")],
                        style=sectionStyle
                    )
                ], className="six columns"),

            ], className="row")
        ])
    elif value == 3:
        tab_layout = html.Div([
            html.Br([]),
            html.Br([]),
            html.Br([]),
            html.Div([
                html.Section(
                    [html.Strong('References'),
                    html.Hr([]),
                    dcc.Markdown(s(
                    '''
                    - [UCI KDD Cup 1999 Page](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
                    - [Tavallaee M., Bagheri, E., Lu, W., Ghorbani, A.A. 2009,
                    "A detailed Analysis of the KDD Cup 99 Data Set",
                    Proceedings of the IEEE Symposium on Computational
                    Intelligence in Security and Defence Applications](http://ieeexplore.ieee.org/document/5356528/)
                    - [Scikit-learn](http://scikit-learn.org/stable/)
                    '''
                    ))],
                    style=sectionStyle
                )
            ], className="twelve columns"),

            html.Div([
                html.Div([
                    html.Section(
                        [html.Strong('Plotly'),
                        html.Hr([]),
                        dcc.Markdown(s(
                        '''
                        - [Plotly]()
                        - [Dash](https://plot.ly/products/dash/)
                        - [Dash Gallery](https://dash.plot.ly/gallery)
                        - [Dash Documentation](https://dash.plot.ly/)
                        - [Hands-on Dash Workshops](https://plotcon.plot.ly/workshops)
                        - [Plotly Python API Documentation](https://plot.ly/python/)
                        - [Plotly Community](https://community.plot.ly/)
                        - [Plotly Webinars](https://plotlywebinars.formulated.by/)
                        '''))],
                        style=sectionStyle
                    )
                ], className='six columns'),

                html.Div([
                    html.Section(
                        [html.Strong('Datascience.com'),
                        html.Hr([]),
                        dcc.Markdown(s(
                        '''
                        - [Datascience.com](https://www.datascience.com/)
                        - [Datascience.com Webinars](https://datascience.hubs.vidyard.com/categories/webinars)
                        '''))],
                        style=sectionStyle
                    )
                ], className='six columns')
            ], className="row")
        ])

    return tab_layout

@app.callback(
    dash.dependencies.Output('frequency-graph', 'figure'),
    [dash.dependencies.Input('frequency-dropdown', 'value')])
def update_graph(value):
    if value == 'Bar':
        fig = chart_attack_counts(df1)
    if value == 'Pie':
        fig = chart_attack_counts(df1, chart_type='pie')
    return fig

@app.callback(
    Output('confusion-matrix', 'figure'),
    [Input('run-button', 'n_clicks')])
def update_graph(n_clicks):
    if n_clicks == 0:
        data = []

        layout = go.Layout(title='Confusion Matrix of Classifier',
                           xaxis={'title':"Predicted Class"},
                           yaxis={'title':"Observed Class"},
                           plot_bgcolor = 'rgb(30, 30, 30)',
                           paper_bgcolor = 'rgb(30, 30, 30)',
                           font = dict(
                               color = textColor
                           ),
                           margin = dict(
                               t = 60, b = 60, l = 60, r = 20
                           ))

        fig = go.Figure(data=data, layout=layout)
    if n_clicks > 0:
        tree = joblib.load('tree_model.sav')

        cv = joblib.load('cv_model.sav')

        cm = confusion_matrix(y_test, cv.best_estimator_.predict(X_test))

        data = [go.Heatmap(z=cm/np.sum(cm,axis=1)
                   , y=tree.classes_
                   , x=tree.classes_
                   , colorscale='Blues'
                   , xaxis='hello'
                  )]

        layout = go.Layout(title='Confusion Matrix of Classifier',
                           xaxis={'title':"Predicted Class"},
                           yaxis={'title':"Observed Class"},
                           plot_bgcolor = 'rgb(30, 30, 30)',
                           paper_bgcolor = 'rgb(30, 30, 30)',
                           font = dict(
                               color = textColor
                           ),
                           margin = dict(
                               t = 60, b = 60, l = 60, r = 20
                           ))

        fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    Output('importance-bar', 'figure'),
    [Input('run-button', 'n_clicks')])
def update_graph(n_clicks):
    if n_clicks == 0:
        data = []
        layout = go.Layout(title='Classifer Feature Importances',
                           xaxis=dict(title='Feature Importance'),
                           yaxis=dict(title='Feature Name'),
                           plot_bgcolor = 'rgb(30, 30, 30)',
                           paper_bgcolor = 'rgb(30, 30, 30)',
                           font = dict(
                               color = textColor
                           ),
                           margin = dict(
                               t = 60, b = 60, l = 60, r = 20
                           ))

        fig = go.Figure(data=data, layout=layout)
    if n_clicks > 0:
        tree = joblib.load('tree_model.sav')

        cv = joblib.load('cv_model.sav')

        classes = cv.best_estimator_.classes_
        class_i = {j: i for i, j in enumerate(classes)}
        interpreter = skater.Interpretation(X, feature_names=features)

        model = skater.model.InMemoryModel(cv.best_estimator_.predict_proba,
                                           examples=X[:5], target_names=classes)

        importances = interpreter.feature_importance.feature_importance(model)
        importances_ = (importances[importances > .01]).reset_index()
        importances_.columns = ['feature','importance']


        # Creating a plotly horizontal bar chart :
        data = [go.Bar(x=importances_['importance'],
                      y=importances_['feature'],
                      orientation='h',
                      marker=dict(color=traceColors[4]))]

        layout = go.Layout(title='Classifer Feature Importances',
                           xaxis=dict(
                                title='Feature Importance',
                           ),
                           yaxis=dict(
                                title='Feature Name',
                                automargin=True
                           ),
                           plot_bgcolor = 'rgb(30, 30, 30)',
                           paper_bgcolor = 'rgb(30, 30, 30)',
                           font = dict(
                               color = textColor
                           ),
                           margin = dict(
                               t = 60, b = 60, l = 60, r = 20
                           ))

        fig = go.Figure(data=data, layout=layout)
    return fig

app.css.append_css({"external_url": "https://codepen.io/bcd/pen/qYOYoB.css"})

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
