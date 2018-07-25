# terrorist-classifier
The goal of this classifier is to predict the terrorist group responsible for an attack based on information about the incident, such as the weapons used and groups targeted.

# Data Source
Terrorist attack data from 1970-2017 were collected from the Global Terrorism Database (GTD), available for download here: https://www.start.umd.edu/gtd/contact/.

The Excel spreadsheet was converted to a .csv file to expedite handling with pandas (see [notebook](Terrorist%20Classifier.ipynb)).

# Running classifier
We chose to implement an MLP neural network to model this supervised classification problem, as we do not have prior knowledge of each feature's statistical power.

You can train the classifier using 75% of the available data, and predict the terrorist groups responsible for the attacks in the test set:
```python
import classifier
classifier.classify(group_thres=5, cate_thres=100)
```
You can also perform k-fold cross-validation on the model:
```python
classifier.cross_validate(group_thres=5, cate_thres=100, k=4)
```
The two variables passed into the classifier are hyperparameters, which determine the level of detail retained in the training set. `group_thres` indicates the minimum number of attacks for a terrorist group to be included in the training set. `cate_thres` sets the threshold of feature instances below which a value will be aggregated.

# Notebook
The classifier is fully documented in the [Jupyter notebook](Terrorist%20Classifier.ipynb), which includes data filtering methodology, hyperparameter selection, and cross-validation results.
