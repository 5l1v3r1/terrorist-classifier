# terrorist-classifier
The goal of this classifier is to predict the terrorist group responsible for an attack based on information about the incident, such as the weapons used and groups targeted.

# Data Source
Terrorist attack data from 1970-2017 were collected from the Global Terrorism Database (GTD), available for download here: https://www.start.umd.edu/gtd/contact/.

The Excel spreadsheet was converted to a .csv file to expedite handling with pandas, using a Python package:

```
pip install xlsx2csv
xlsx2csv ./globalterrorismdb_0617dist.xlsx ./globalterrorismdb_0617dist.csv
```
```
import os
os.chdir('/your directory/')
import pandas as pd
df_full = pd.read_csv('GTD_0617dist/globalterrorismdb_0617dist.csv')
```

# Running classifier
You can train the classifier and predict terrorist groups in the test set as shown below. 
```
import classifier
classifier.classify(group_thres=5, cate_thres=100)
```
The two variables passed into the classifier are hyperparameters for the model, which dictate the size of the training set. group_thres indicates the minimum number of attacks for a terrorist group to be included in the training set. cate_thres sets the threshold of feature instances below which a value will be aggregated. 
