## Mooring DBSCAN v2

## Running the module
Running the module can be done by executing it with simple python code `python mooring_dbscan.py`.
This executes all the parts of the pipeline. 

Individual parts of the pipeline can be executed if so desired.
### Preprocessing
The preprocessing part of the module reads the data from either a local csv file or a MonetDB database instance. The 
preprocessing filters the data and calculates the aggregate data to be used in model training.

Running this step individually requires the user to import the DataObject object and executing it as follows

```
from mooring_dbscan import DataObject

dataobj = DataObject()
dataobj.run()
```
This will run all the preprocessing steps and the data will be available in `dataobj.train`

### Modeling
The modeling object takes the training data created by the preprocessing and detects clusters from it. The modeling 
object then creates polygons out of the cluster points based on the dimensions of ships that form the cluster point. The 
modeling object also detects quay areas that are formed from multiple individual berths.

Running this step individually requires the user to import the Model object and executing it as follows

```
from mooring_dbscan import Model

modelobj = Model()
modelobj.run()
```
The detected berths will be available in `model.clustering_results` and the detected quays in `model.combined_quays`


### Visualizer

The code includes a visualizer that maps the found polygons to a folium map. The map is saved to a HTML file.

Running this step individually requires the user to import the Visualizer model and executing it as follows

```
from mooring_dbscan import Visualizer

Visualizer(modelobj)
```
This will map the berths and quays detected by `modelobj` and store the results to an HTML file.

## Config 

The execution of the code can be configured using the `config.py` file.

Most important configuration variables are:

LOCAL_FILE: the name of the local file. If missing or left blank, the data will be fetched from a database

HTML_FILE: the name of the html file where the results will be saved.