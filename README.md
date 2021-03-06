![iot image](https://camo.githubusercontent.com/56fb7c9e3129755320229df81eea6d1a4635d468/68747470733a2f2f63646e322e68756273706f742e6e65742f68756266732f3533323034352f706c6f746c792d776562696e61722d696f742d6e6574776f726b2d696e74727573696f6e2d393030783435302e6a7067)

![plotly_ds_image](https://user-images.githubusercontent.com/5395649/39215767-f1380216-47cd-11e8-8db2-c909810bbb65.png)

# Detecting IoT Network Intrusion With Plotly

Repo containing supporting material of the webinar entitled "[Detecting IoT Networking Intrusion with Plotly](https://www.datascience.com/resources/webinars/plotly-iot-network-intrusion)" on April 25, 2018

**You want to run these notebooks on the [DataScience.com Platform](http://www.datascience.com)? Request a demo of the Platform [here](https://www.datascience.com/request-demo?hsCtaTracking=74a4e9d5-d6c1-4dc1-b494-9f4c45847000%7C4e0d2014-a652-446c-b385-da9d9219ba70).**

The notebooks shared in this repo have been developed by [Aaron Kramer](https://github.com/aikramer2)
with contributions from Jean-Rene Gauthier at DataScience.com.

## Installation

Installation instructions can be found in the `environment/` folder. We highly
recommend pulling a pre-built Docker image we created for this webinar.

## In this Repo

You will find the following notebooks:

* `exploratory-data-analysis.ipynb`: We use Plotly's Python API to explore the KDD Cup 99 dataset.

* `intrusion-classification-model-build.ipynb`: Following data exploration, we develop a few models to identify potential attacks from normal connections. We interpret how the different models make their classification decisions using the model interpretation library [Skater](https://github.com/datascienceinc/Skater)

* `dash-app.py`: An interactive Dash app that displays graphs and tables using the same models that were developed in the Jupyter Notebooks. Run this app with `$ python dash-app.py`. Learn more about Dash in the [Dash User Guide](https://dash.plot.ly).

There is also a `utilities/` folder that contains a series of utility functions that are used in both Jupyter notebooks.
