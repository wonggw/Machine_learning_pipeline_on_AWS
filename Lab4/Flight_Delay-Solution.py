#!/usr/bin/env python
# coding: utf-8

# # Project Solution Lab 2: 

# # Problem: Predicting Airplane Delays
# 
# The multi-fold goals of this notebook are:
# - Process and create a dataset from downloaded ZIP files
# - Exploratory data analysis (EDA)
# - Establish a baseline model
# - Move from a simple model to an ensemble model
# - Hyperparameter optimization
# - Check feature importance
# 
# ## Introduction to business scenario
# You work for a travel booking website that is working to improve the customer experience for flights that were delayed. The company wants to create a feature to let customers know if the flight will be delayed due to weather when the customers are booking the flight to or from the busiest airports for domestic travel in the US. 
# 
# You are tasked with solving part of this problem by leveraging machine learning to identify whether the flight will be delayed due to weather. You have been given access to the a dataset of on-time performance of domestic flights operated by large air carriers. You can use this data to train a machine learning model to predict if the flight is going to be delayed for the busiest airports.
# 
# ## About this dataset
# This dataset contains scheduled and actual departure and arrival times reported by certified US air carriers that account for at least 1% of domestic scheduled passenger revenues. The data was collected by the Office of Airline Information, Bureau of Transportation Statistics (BTS). The dataset contains date, time, origin, destination, airline, distance, and delay status of flights for flights between 2013 and 2018.
# 
# ### Features
# For more information about features in the dataset, see [On-time delay dataset features](https://www.transtats.bts.gov/Fields.asp).
# 
# ### Dataset attributions  
# Website: https://www.transtats.bts.gov/
# 
# Dataset(s) used in this lab were compiled by the Office of Airline Information, Bureau of Transportation Statistics (BTS), Airline On-Time Performance Data, available at https://www.transtats.bts.gov/DatabaseInfo.asp?DB_ID=120&DB_URL=Mode_ID=1&Mode_Desc=Aviation&Subject_ID2=0.

# # Step 1: Problem formulation and data collection
# 
# Start this project off by writing a few sentences below that summarize the business problem and the business goal you're trying to achieve in this scenario. Include a business metric you would like your team to aspire toward. With that information defined, clearly write out the machine learning problem statement. Finally, add a comment or two about the type of machine learning this represents. 
# 
# #### <span style="color: blue;">Project presentation: Include a summary of these details in your project presentations.</span>
# 
# 
# ### 1. Determine if and why ML is an appropriate solution to deploy.
# \# Write your answer here
# 
# ### 2. Formulate the business problem, success metrics, and desired ML output.
# \# Write your answer here
# 
# ### 3. Identify the type of ML problem you’re dealing with.
# \# Write your answer here
# 
# ### 4. Analyze the appropriateness of the data you’re working with.
# \# Write your answer here

# ### Setup
# 
# Now that we have decided where to focus our energy, let's set things up so you can start working on solving the problem.
# 
# **Note:** This notebook was created and tested on an `ml.m4.xlarge` notebook instance. 

# In[1]:


import os
from pathlib2 import Path
from zipfile import ZipFile
import time

import pandas as pd
import numpy as np
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Step 2: Data preprocessing and visualization
# In this data preprocessing phase, you should take the opportunity to explore and visualize your data to better understand it. First, import the necessary libraries and read the data into a Pandas dataframe. After that, explore your data. Look for the shape of the dataset and explore your columns and the types of columns you're working with (numerical, categorical). Consider performing basic statistics on the features to get a sense of feature means and ranges. Take a close look at your target column and determine its distribution.
# 
# ### Specific questions to consider
# 1. What can you deduce from the basic statistics you ran on the features? 
# 
# 2. What can you deduce from the distributions of the target classes?
# 
# 3. Is there anything else you deduced from exploring the data?
# 
# #### <span style="color: blue;">Project presentation: Include a summary of your answers to these and other similar questions in your project presentations.</span>

# Start by bringing in the dataset from an Amazon S3 public bucket to this notebook environment.

# In[2]:


# Check whether the file is already in the desired path or if it needs to be downloaded

base_path = '/home/ec2-user/SageMaker/project/data/FlightDelays/'
csv_base_path = '/home/ec2-user/SageMaker/project/data/csvFlightDelays/'
file_path = 'On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2014_1.zip'

if not os.path.isfile(base_path + file_path):
    subprocess.run(['mkdir', '-p', base_path])
    subprocess.run(['mkdir', '-p', csv_base_path])
    subprocess.run(['aws', 's3', 'cp', 
                    's3://aws-tc-largeobjects/ILT-TF-200-MLDWTS/flight_delay_project/csvFlightData-5/', 
                    base_path,'--recursive'])
else:
    print('File already downloaded!')


# In[3]:


zip_files = [str(file) for file in list(Path(base_path).iterdir()) if '.zip' in str(file)]
len(zip_files)


# #### Extract CSV files from ZIP files

# In[4]:


def zip2csv(zipFile_name , file_path = '/home/ec2-user/SageMaker/project/data/csvFlightDelays'):
    """
    Extract csv from zip files
    zipFile_name: name of the zip file
    file_path : name of the folder to store csv
    """
    try:
        with ZipFile(zipFile_name, 'r') as z: 
            print(f'Extracting {zipFile_name} ') 
            z.extractall(path=file_path) 
    except:
        print(f'zip2csv failed for {zipFile_name}')

for file in zip_files:
    zip2csv(file)

print("Files Extracted")


# In[5]:


csv_files = [str(file) for file in list(Path(csv_base_path).iterdir()) if '.csv' in str(file)]
len(csv_files)


# Before loading the CSV file, read the HTML file from the extracted folder. This HTML file includes the background and more information on the features included in the dataset.

# In[6]:


from IPython.display import IFrame, HTML

IFrame(src="./data/csvFlightDelays/readme.html", width=1000, height=600)


# #### Load sample CSV
# 
# Before combining all the CSV files, get a sense of the data from a single CSV file. Using Pandas, read the `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2018_9.csv` file first. You can use the Python built-in `read_csv` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)).

# In[7]:


df_temp = pd.read_csv(f"{csv_base_path}On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2018_9.csv")


# **Question**: Print the row and column length in the dataset, and print the column names.
# 
# **Hint**: Use the `<dataframe>.shape` function to view the rows and columns of a dataframe and `<dataframe>.columns` to view the column names.

# In[8]:


df_shape = df_temp.shape
print(f'Rows and columns in one csv file is {df_shape}')


# **Question**: Print the first 10 rows of the dataset.  
# 
# **Hint**: Use the built-in Pandas function `head(x)` to print `x` number of rows.

# In[9]:


df_temp.head(10)


# **Question**: Print all the columns in the dataset. Use `<dataframe>.columns` to view the column names.

# In[10]:


print(f'The column names are :')
print('#########')
for col in df_temp.columns:
    print(col)


# **Question**: Print all the columns in the dataset that contain the word 'Del'. This will help you see how many columns have delay data in them.
# 
# **Hint**: You can use a Python list comprehension to include values that pass certain `if` statement criteria.
# 
# For example: `[x for x in [1,2,3,4,5] if x > 2]`  
# 
# **Hint**: You can use the `in` keyword ([documentation](https://www.w3schools.com/python/ref_keyword_in.asp)) to check if the value is in a list or not. 
# 
# For example: `5 in [1,2,3,4,5]`

# In[11]:


[col for col in df_temp.columns if 'Del' in col]


# Here are some more questions to help you find out more about your dataset.
# 
# **Questions**   
# 1. How many rows and columns does the dataset have?   
# 2. How many years are included in the dataset?   
# 3. What is the date range for the dataset?   
# 4. Which airlines are included in the dataset?   
# 5. Which origin and destination airports are covered?

# In[12]:


print("The #rows and #columns are ", df_temp.shape[0] , " and ", df_temp.shape[1])
print("The years in this dataset are: ", list(df_temp.Year.unique()))
print("The months covered in this dataset are: ", sorted(list(df_temp.Month.unique())))
print("The date range for data is :" , min(df_temp.FlightDate), " to ", max(df_temp.FlightDate))
print("The airlines covered in this dataset are: ", list(df_temp.Reporting_Airline.unique()))
print("The Origin airports covered are: ", list(df_temp.Origin.unique()))
print("The Destination airports covered are: ", list(df_temp.Dest.unique()))


# **Question**: What is the count of all the origin and destination airports?
# 
# **Hint**: You can use the Pandas `values_count` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html)) to find out the values for each airport using the columns `Origin` and `Dest`.

# In[13]:


counts = pd.DataFrame({'Origin':df_temp['Origin'].value_counts(), 'Destination':df_temp['Dest'].value_counts()})
counts


# **Question**: Print the top 15 origin and destination airports based on number of flights in the dataset.
# 
# **Hint**: You can use the Pandas `sort_values` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)).

# In[14]:


counts.sort_values(by=['Origin'],ascending=False).head(15)


# **Question**: Given all the information about a flight trip, can you predict if it would be delayed?

# **Answer**: The column **ArrDel15** is an indicator variable that takes the value 1 when the delay is more than 15 minutes, 0 otherwise.
# 
# You could use this as a target column for the classification problem.

# Now, assume you are traveling from San Francisco to Los Angeles on a work trip. You want to have an ideas if your flight will be delayed, given a set of features, so that you can manage your reservations in Los Angeles better. How many features from this dataset would you know before your flight?
# 
# Columns such as `DepDelay`, `ArrDelay`, `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`, and `DivArrDelay` contain information about a delay. But this delay could have occured at the origin or destination. If there were a sudden weather delay 10 minutes before landing, this data would not be helpful in managing your Los Angeles reservations.
# 
# So to simplify the problem statement, consider the following columns to predict an arrival delay:<br>
# 
# `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`, `FlightDate`, `Reporting_Airline`, `Origin`, `OriginState`, `Dest`, `DestState`, `CRSDepTime`, `DepDelayMinutes`, `DepartureDelayGroups`, `Cancelled`, `Diverted`, `Distance`, `DistanceGroup`, `ArrDelay`, `ArrDelayMinutes`, `ArrDel15`, `AirTime`
# 
# You will also filter the source and destination airports to be:
# - Top airports: ATL, ORD, DFW, DEN, CLT, LAX, IAH, PHX, SFO
# - Top 5 airlines: UA, OO, WN, AA, DL
# 
# This should help in reducing the size of data across the CSV files to be combined.

# #### Combine all CSV files
# 
# **Hint**:  
# First, create an empy dataframe that you will use to copy your individual dataframes from each file. Then, for each file in the `csv_files` list:
# 
# 1. Read the CSV file into a dataframe  
# 2. Filter the columns based on the `filter_cols` variable
# 
# ```
#         columns = ['col1', 'col2']
#         df_filter = df[columns]
# ```
# 
# 3. Keep only the subset_vals in each of the subset_cols. Use the `isin` Pandas function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html)) to check if the `val` is in the dataframe column and then choose the rows that include it.
# 
# ```
#         df_eg[df_eg['col1'].isin('5')]
# ```
# 
# 4. Concatenate the dataframe with the empty dataframe 

# In[15]:


def combine_csv(csv_files, filter_cols, subset_cols, subset_vals, file_name = 'data/combined_files.csv'):
    """
    Combine csv files into one Data Frame
    csv_files: list of csv file paths
    filter_cols: list of columns to filter
    subset_cols: list of columns to subset rows
    subset_vals: list of list of values to subset rows
    """
    
    df = pd.DataFrame()
    
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_temp = df_temp[filter_cols]
        for col, val in zip(subset_cols,subset_vals):
            df_temp = df_temp[df_temp[col].isin(val)]      
        
        df = pd.concat([df, df_temp], axis=0)
    
        
    df.to_csv(file_name, index=False)
    print(f'Combined csv stored at {file_name}')


# In[16]:


#cols is the list of columns to predict Arrival Delay 
cols = ['Year','Quarter','Month','DayofMonth','DayOfWeek','FlightDate',
        'Reporting_Airline','Origin','OriginState','Dest','DestState',
        'CRSDepTime','Cancelled','Diverted','Distance','DistanceGroup',
        'ArrDelay','ArrDelayMinutes','ArrDel15','AirTime']

subset_cols = ['Origin', 'Dest', 'Reporting_Airline']

# subset_vals is a list collection of the top origin and destination airports and top 5 airlines
subset_vals = [['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['ATL', 'ORD', 'DFW', 'DEN', 'CLT', 'LAX', 'IAH', 'PHX', 'SFO'], 
               ['UA', 'OO', 'WN', 'AA', 'DL']]


# Use the function above to merge all the different files into a single file that you can read easily. 
# 
# **Note**: This will take 5-7 minutes to complete.

# In[17]:


start = time.time()
combine_csv(csv_files, cols, subset_cols, subset_vals)
print(f'csv\'s merged in {round((time.time() - start)/60,2)} minutes')


# #### Load dataset
# 
# Load the combined dataset.

# In[18]:


data = pd.read_csv('/home/ec2-user/SageMaker/project/data/combined_files.csv')


# Print the first 5 records.

# In[19]:


data.head(5)


# Here are some more questions to help you find out more about your dataset.
# 
# **Questions**   
# 1. How many rows and columns does the dataset have?   
# 2. How many years are included in the dataset?   
# 3. What is the date range for the dataset?   
# 4. Which airlines are included in the dataset?   
# 5. Which origin and destination airports are covered?

# In[20]:


print("The #rows and #columns are ", data.shape[0] , " and ", data.shape[1])
print("The years in this dataset are: ", list(data.Year.unique()))
print("The months covered in this dataset are: ", sorted(list(data.Month.unique())))
print("The date range for data is :" , min(data.FlightDate), " to ", max(data.FlightDate))
print("The airlines covered in this dataset are: ", list(data.Reporting_Airline.unique()))
print("The Origin airports covered are: ", list(data.Origin.unique()))
print("The Destination airports covered are: ", list(data.Dest.unique()))


# Let's define our **target column : is_delay** (1 - if arrival time delayed more than 15 minutes, 0 - otherwise). Use the `rename` method to rename the column from `ArrDel15` to `is_delay`.
# 
# **Hint**: You can use the Pandas `rename` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html)).
# 
# For example:
# ```
# df.rename(columns={'col1':'column1'}, inplace=True)
# ```

# In[21]:


# Rename ArrDel15 to is_delay
data.rename(columns={'ArrDel15':'is_delay'}, inplace=True) 


# Look for nulls across columns. You can use the `isnull()` function ([documentation](https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.isnull.html)).
# 
# **Hint**: `isnull()` detects whether the particular value is null or not and gives you a boolean (True or False) in its place. Use the `sum(axis=0)` function to sum up the number of columns.

# In[22]:


data.isnull().sum(axis = 0)


# The arrival delay details and airtime are missing for 22540 out of 1658130 rows, which is 1.3%. You can either remove or impute these rows. The documentation does not mention anything about missing rows.
# 
# **Hint**: Use the `~` operator to choose the values that aren't null from the `isnull()` output.
# 
# For example:
# ```
# null_eg = df_eg[~df_eg['column_name'].isnull()]
# ```

# In[23]:


### Remove null columns
data = data[~data.is_delay.isnull()]
data.isnull().sum(axis = 0)


# Get the hour of the day in 24-hour time format from CRSDepTime.

# In[24]:


data['DepHourofDay'] = (data['CRSDepTime']//100)


# ## **The ML problem statement**
# - Given a set of features, can you predict if a flight is going to be delayed more than 15 minutes?
# - Because the target variable takes only 0/1 value, you could use a classification algorithm. 
# 
# Before jumping to modeling, it is always a good practice to look at feature distribution, correlations, etc.
# - This will give an idea of any non-linearity/patterns in the data.
#     - Linear models: Add power/exponential/interaction features
#     - Try a non-linear model
# - Data imbalance 
#     - Choose metrics that will not give biased model performance (accuracy. vs AUC)
#     - Use weighted/custom loss functions
# - Missing data
#     - Do imputation based on simple statistics - mean, median, mode (numerical variables) frequent class (categorical variables)
#     - Clustering based imputation (KNNs to predict column value)
#     - Drop column

# ### Data exploration
# 
# #### Check class delay vs. no delay
# 
# **Hint**: Use a `groupby` plot ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)) with a `bar` plot ([documentation](https://matplotlib.org/tutorials/introductory/pyplot.html)) to plot the frequency vs. distribution of the class.

# In[25]:


(data.groupby('is_delay').size()/len(data) ).plot(kind='bar')
plt.ylabel('Frequency')
plt.title('Distribution of classes')
plt.show()


# **Question**: What can you deduce from the bar plot about the ratio of delay vs. no delay?

# **Answer:** 80% of flights arrive on time, but 20% do not. This is clear class imbalance.

# **Questions**: 
# 
# - Which months have the most delays?
# - What time of the day has the most delays?
# - What day of the week has the most delays?
# - Which airline has the most delays?
# - Which origin and destination airports have the most delays?
# - Is flight distance a factor in the delays?

# In[26]:


viz_columns = ['Month', 'DepHourofDay', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest']
fig, axes = plt.subplots(3, 2, figsize=(20,20), squeeze=False)
# fig.autofmt_xdate(rotation=90)

for idx, column in enumerate(viz_columns):
    ax = axes[idx//2, idx%2]
    temp = data.groupby(column)['is_delay'].value_counts(normalize=True).rename('percentage').    mul(100).reset_index().sort_values(column)
    sns.barplot(x=column, y="percentage", hue="is_delay", data=temp, ax=ax)
    plt.ylabel('% delay/no-delay')
    

plt.show()


# In[27]:


sns.lmplot( x="is_delay", y="Distance", data=data, fit_reg=False, hue='is_delay', legend=False)
plt.legend(loc='center')
plt.xlabel('is_delay')
plt.ylabel('Distance')
plt.show()


# **Answers**
# - We see that the % delay is high in May, June, July, and August (Months 5-8).
# - % delays are higher at night than during the day. There are no flights departing at 3AM!
# - Sunday(1), Wednesday(4), Saturday(7) have relatively higher delays than other days.
# - Airline WN has the highest % delays compared to others.
# - Delays seem to occur in both short and long distance flight journeys.
# - Flights with SFO as a destination have higher delays than others.
# 
# Even though some of the variables are encoded in numbers, it is important to consider them as categorical variables because we don't see a quantitative relation between them and the target variable.

# ### Features
# 
# Look at all the columns and what their specific types are.

# In[28]:


data.columns


# Filtering the required columns:
# - Date is redundant, because you have Year, Quarter, Month, DayofMonth, and DayOfWeek to describe the date.
# - Use Origin and Dest codes instead of OriginState and DestState.
# - Because you are just classifying whether the flight is delayed or not, you don't need TotalDelayMinutes, DepDelayMinutes, and ArrDelayMinutes.
# 
# Treat DepHourofDay as a categorical variable because it doesn't have any quantitative relation with the target.
# - If you had to do a one-hot encoding of it, it would result in 23 more columns.
# - Other alternatives to handling categorical variables include hash encoding, regularized mean encoding, and bucketizing the values, among others.
# - Just split into buckets here.
# 
# **Hint**: To change a column type to category, use the `astype` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html)).

# In[29]:


data_orig = data.copy()
data = data[[ 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest','Distance','DepHourofDay', 'is_delay']]
categorical_columns  = ['Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest', 'DepHourofDay', 'is_delay']
for c in categorical_columns:
    data[c] = data[c].astype('category')


# To use one-hot encoding, use the Pandas `get_dummies` function for the categorical columns that you selected above. Then, you can concatenate those generated features to your original dataset using the Pandas `concat` function. For encoding categorical variables, you can also use *dummy encoding* by using a keyword `drop_first=True`. For more information on dummy encoding, see https://en.wikiversity.org/wiki/Dummy_variable_(statistics).
# 
# For example:
# ```
# pd.get_dummies(df[['column1','columns2']], drop_first=True)
# ```

# In[30]:


categorical_columns.remove('is_delay')
data_dummies = pd.get_dummies(data[categorical_columns], drop_first=True)
data = pd.concat([data, data_dummies], axis = 1)
data.drop(categorical_columns,axis=1, inplace=True)


# Check the length of the dataset and the new columnms.

# In[31]:


len(data)


# In[32]:


data.columns


# Now you are ready to do model training. Before splitting the data, rename the column `is_delay` to `target`.
# 
# **Hint**: You can use the Pandas `rename` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html)).

# In[33]:


data.rename(columns = {'is_delay':'target'}, inplace=True )


# # Project Solution Lab 3: 

# # Step 3: Model training and evaluation
# 
# There are some preliminary steps that you must include when converting the dataset from a dataframe to a format that a machine learning algorithm can use. For Amazon SageMaker, here are the steps you need to take:
# 
# 1. Split the data into `train_data`, `validation_data`, and `test_data` using `sklearn.model_selection.train_test_split`.  
# 2. Convert the dataset to an appropriate file format that the Amazon SageMaker training job can use. This can be either a CSV file or record protobuf. For more information, see [Common Data Formats for Training](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html).  
# 3. Upload the data to your Amazon S3 bucket. If you have not created one before, see [Create a Bucket](https://docs.aws.amazon.com/AmazonS3/latest/gsg/CreatingABucket.html).  
# 
# Use the following cells to complete these steps. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Take note of the key decisions you've made in this phase in your project presentation.</span>

# ### Train test split

# In[34]:


from sklearn.model_selection import train_test_split

def create_training_sets(data):
    """
    Convert data frame to train, validation and test
    params:
        data: The dataframe with the dataset to be split
    Returns:
        train_features: Training feature dataset
        test_features: Test feature dataset 
        train_labels: Labels for the training dataset
        test_labels: Labels for the test dataset
        val_features: Validation feature dataset
        val_labels: Labels for the validation dataset
    """
    # Extract the target variable from the dataframe and convert the type to float32
    ys = np.array(data["target"]).astype("float32")
    
    # Drop all the unwanted columns including the target column
    drop_list = ["target"]
    
    # Drop the columns from the drop_list and convert the data into a NumPy array of type float32
    xs = np.array(data.drop(drop_list, axis=1)).astype("float32")
    
    np.random.seed(0)
    
    # Use the sklearn function train_test_split to split the dataset in the ratio train 80% and test 20%
    train_features, test_features, train_labels, test_labels = train_test_split(
    xs, ys, test_size=0.2)
    
    # Use the sklearn function again to split the temp dataset into 50% validation and 50% test
    val_features, test_features, val_labels, test_labels = train_test_split(
    test_features, test_labels, test_size=0.5)
    
    return train_features, test_features, train_labels, test_labels, val_features, val_labels


# In[35]:


train_features, test_features, train_labels, test_labels, val_features, val_labels = create_training_sets(data)

print(f"Length of train_features is: {train_features.shape}")
print(f"Length of train_labels is: {train_labels.shape}")
print(f"Length of val_features is: {val_features.shape}")
print(f"Length of val_labels is: {val_labels.shape}")
print(f"Length of test_features is: {test_features.shape}")
print(f"Length of test_labels is: {test_labels.shape}")


# **Answer**
# ```
# Length of train_features is: (1308472, 94)
# Length of train_labels is: (1308472,)
# Length of val_features is: (163559, 94)
# Length of val_labels is: (163559,)
# Length of test_features is: (163559, 94)
# Length of test_labels is: (163559,)
# ```

# ### Baseline classification model

# In[36]:


import sagemaker
from sagemaker.predictor import csv_serializer
from sagemaker.amazon.amazon_estimator import RecordSet
import boto3

# Instantiate the LinearLearner estimator object
num_classes = len(pd.unique(train_labels)) 

classifier_estimator = sagemaker.LinearLearner(role=sagemaker.get_execution_role(),
                                               instance_count=1,
                                               instance_type='ml.m4.xlarge',
                                               predictor_type='binary_classifier',
                                              binary_classifier_model_selection_criteria = 'cross_entropy_loss')


# Linear learner accepts training data in protobuf or CSV content types, and accepts inference requests in protobuf, CSV, or JSON content types. Training data has features and ground-truth labels, while the data in an inference request has only features.
# 
# In a production pipeline, it is recommended to convert the data to the Amazon SageMaker protobuf format and store it in Amazon S3. However, to get up and running quickly, AWS provides the convenient method `record_set` for converting and uploading when the dataset is small enough to fit in local memory. It accepts NumPy arrays like the ones you already have, so let's use it here. The `RecordSet` object will keep track of the temporary Amazon S3 location of your data. Use the `estimator.record_set` function to create train, validation, and test records. Then, use the `estimator.fit` function to start your training job.

# In[37]:


### Create train, val, test record
train_records = classifier_estimator.record_set(train_features, train_labels, channel='train')
val_records = classifier_estimator.record_set(val_features, val_labels, channel='validation')
test_records = classifier_estimator.record_set(test_features, test_labels, channel='test')

### Fit the classifier
classifier_estimator.fit([train_records, val_records, test_records])


# ## Model evaluation
# In this section, you'll evaluate your trained model. First, use the `estimator.deploy` function with `initial_instance_count= 1` and `instance_type= 'ml.m4.xlarge'` to deploy your model on Amazon SageMaker.

# In[38]:


### Deloy an endpoint for batch prediction
classifier_predictor = classifier_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# Once your endpoint is 'InService', evaluate how your model performs on the test set. Use the `predict_batches` function to predict the metrics on your test set.

# In[39]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

def predict_batches(predictor, features, labels):
    """
    Return evaluation results
    predictor : Predictor object of model
    features: Input features to model
    label: Ground truth target values
    """
    prediction_batches = [predictor.predict(batch) for batch in np.array_split(features, 100)]

    # Parse protobuf responses to extract predicted labels
    extract_label = lambda x: x.label['predicted_label'].float32_tensor.values
    preds = np.concatenate([np.array([extract_label(x) for x in batch]) for batch in prediction_batches])
    preds = preds.reshape((-1,))

    # Calculate accuracy
    accuracy = (preds == labels).sum() / labels.shape[0]
    print(f'Accuracy: {accuracy}')
    
    auc = roc_auc_score(labels, preds)
    print(f'AUC     : {auc}')
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, preds, average = 'binary')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1_score: {f1_score}')
    
    confusion_matrix = pd.crosstab(index=labels, columns=np.round(preds), rownames=['True'], colnames=['predictions']).astype(int)
    plt.figure(figsize = (5,5))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap="YlGnBu").set_title('Confusion Matrix') 


# In[40]:


predict_batches(classifier_predictor, test_features, test_labels)


# ### Key questions to consider:
# 1. How does your model's performance on the test set compare to the training set? What can you deduce from this comparison? 
# 
# 2. Are there obvious differences between the outcomes of metrics like accuracy, precision, and recall? If so, why might you be seeing those differences? 
# 
# 3. Given your business situation and goals, which metric(s) is most important for you to consider here? Why?
# 
# 4. Is the outcome for the metric(s) you consider most important sufficient for what you need from a business standpoint? If not, what are some things you might change in your next iteration (in the feature engineering section, which is coming up next)? 
# 
# Use the cells below to answer these and other questions. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Record answers to these and other similar questions you might answer in this section in your project presentations. Record key details and decisions you've made in your project presentations.</span>

# **Question**: What can you summarize from the confusion matrix?

# #### Case of class imbalance
# - When dealing with imbalanced classes, it is important to weigh in between which metric you can optimize to solve a given problem at hand.  
# - You can choose precision or recall, depending on whether only correct positive predictions matter or all positive predictions are of most importance.  
# - You can also balance between precision and recall with custom weights with F1-beta score.  
# 
# - AUC measures the area under the TPR vs. FPR curve. It is the power of the model to discriminate between a positive and negative prediction.  
# - An AUC of 0.5 means the model predictions are as good as random prediction.  
# - Note that AUC is independent of threshold online precision or recall, making it easier to compare models.  
# 
# For the classifier above, we see that:
# - Accuracy ~ 80% and AUC ~0.5, meaning the model does not have enough predictive power and is as good as a random choice.  
# - High false negatives resulted in a low recall value.  
#     
# To improve classifier performance, you can:
# - Add more features  
# - Optimize for a different metric/loss  
# - Try non-linear models  
# - Do ensemble modeling  
# 
# For more information about customizing linear learner, see [Train Faster, More Flexible Models with Amazon SageMaker Linear Learner](https://aws.amazon.com/blogs/machine-learning/train-faster-more-flexible-models-with-amazon-sagemaker-linear-learner/).

# # Project Solution Lab 4: 

# # Iteration II

# # Step 4: Feature engineering
# 
# You've now gone through one iteration of training and evaluating your model. Given that the outcome you reached for your model the first time probably wasn't sufficient for solving your business problem, what are some things you could change about your data to possibly improve model performance?
# 
# ### Key questions to consider:
# 1. How might the balance of your two main classes (delay and no delay) impact model performance?
# 2. Do you have any features that are correlated?
# 3. Are there feature reduction techniques you could perform at this stage that might have a positive impact on model performance? 
# 4. Can you think of adding some more data/datasets?
# 4. After performing some feature engineering, how does your model performance compare to the first iteration?
# 
# Use the cells below to perform specific feature engineering techniques (per the questions above) that you think could improve your model performance. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Record key decisions and methods you use in this section in your project presentations, as well as any new performance metrics you obtain after evaluating your model again.</span>
# 
# Before you start, think about why the precision and recall are around 80% while the accuracy is 99%.

# #### Add more features
# 
# 1. Holidays
# 2. Weather

# Because the list of holidays from 2014 to 2018 is known, you can create an indicator variable **is_holiday** to mark these.
# The hypothesis is that airplane delays could be higher during holidays compared to the rest of the days. Add a boolean variable `is_holiday` that includes the holidays for the years 2014-2018.

# In[41]:


# Source: http://www.calendarpedia.com/holidays/federal-holidays-2014.html

holidays_14 = ['2014-01-01',  '2014-01-20', '2014-02-17', '2014-05-26', '2014-07-04', '2014-09-01', '2014-10-13', '2014-11-11', '2014-11-27', '2014-12-25' ] 
holidays_15 = ['2015-01-01',  '2015-01-19', '2015-02-16', '2015-05-25', '2015-06-03', '2015-07-04', '2015-09-07', '2015-10-12', '2015-11-11', '2015-11-26', '2015-12-25'] 
holidays_16 = ['2016-01-01',  '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11', '2016-11-24', '2016-12-25', '2016-12-26']
holidays_17 = ['2017-01-02', '2017-01-16', '2017-02-20', '2017-05-29' , '2017-07-04', '2017-09-04' ,'2017-10-09', '2017-11-10', '2017-11-23', '2017-12-25']
holidays_18 = ['2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28' , '2018-07-04', '2018-09-03' ,'2018-10-08', '2018-11-12','2018-11-22', '2018-12-25']
# holidays_19 = ['2019-01-01', '2019-01-21', '2019-02-18', '2019-05-27' , '2019-07-04', '2019-09-02' ,'2019-10-14', '2019-11-11','2019-11-28', '2019-12-25']
holidays = holidays_14+ holidays_15+ holidays_16 + holidays_17+ holidays_18

### Add indicator variable for holidays
data_orig['is_holiday'] = data_orig.FlightDate.isin(holidays).astype(int)


# Weather data was fetched from https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00023174,USW00012960,USW00003017,USW00094846,USW00013874,USW00023234,USW00003927,USW00023183,USW00013881&dataTypes=AWND,PRCP,SNOW,SNWD,TAVG,TMIN,TMAX&startDate=2014-01-01&endDate=2018-12-31.
# <br>
# 
# This dataset has information on wind speed, precipitation, snow, and temperature for cities by their airport codes.
# 
# **Question**: Could bad weather due to rains, heavy winds, or snow lead to airplane delay? Let's check!

# In[42]:


get_ipython().run_cell_magic('bash', '', "aws s3 cp s3://aws-tc-largeobjects/ILT-TF-200-MLDWTS/flight_delay_project/daily-summaries.csv /home/ec2-user/SageMaker/project/data/\n#wget 'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00023174,USW00012960,USW00003017,USW00094846,USW00013874,USW00023234,USW00003927,USW00023183,USW00013881&dataTypes=AWND,PRCP,SNOW,SNWD,TAVG,TMIN,TMAX&startDate=2014-01-01&endDate=2018-12-31' -O /home/ec2-user/SageMaker/project/data/daily-summaries.csv")


# Import weather data prepared for the airport codes in our dataset. Use the stations and airports below for the analysis, and create a new column called `airport` that maps the weather station to the airport name.

# In[43]:


weather = pd.read_csv('/home/ec2-user/SageMaker/project/data/daily-summaries.csv')
station = ['USW00023174','USW00012960','USW00003017','USW00094846','USW00013874','USW00023234','USW00003927','USW00023183','USW00013881'] 
airports = ['LAX', 'IAH', 'DEN', 'ORD', 'ATL', 'SFO', 'DFW', 'PHX', 'CLT']

### Map weather stations to airport code
station_map = {s:a for s,a in zip(station, airports)}
weather['airport'] = weather['STATION'].map(station_map)


# Create another column called `MONTH` from the `DATE` column.

# In[44]:


weather['MONTH'] = weather['DATE'].apply(lambda x: x.split('-')[1])
weather.head()


# Analyze and handle the `SNOW` and `SNWD` columns for missing values using `fillna()`. Use the `isna()` function to check the missing values for all the columns.

# In[45]:


weather.SNOW.fillna(0, inplace=True)
weather.SNWD.fillna(0, inplace=True)
weather.isna().sum()


# **Question**: Print the index of the rows that have missing values for TAVG, TMAX, TMIN.
# 
# **Hint**: Use the `isna()` function to find the rows that are missing, and then use the list on the idx variable to get the index.

# In[46]:


idx = np.array([i for i in range(len(weather))])
TAVG_idx = idx[weather.TAVG.isna()]
TMAX_idx = idx[weather.TMAX.isna()]
TMIN_idx = idx[weather.TMIN.isna()]
TAVG_idx


# You can replace the missing TAVG, TMAX, and TMIN with the average value for a particular station/airport. Because the consecutive rows of TAVG_idx are missing, replacing with a previous value would not be possible. Instead, replace it with the mean. Use the `groupby` function to aggregate the variables with a mean value.

# In[47]:


weather_impute = weather.groupby(['MONTH','STATION']).agg({'TAVG':'mean','TMAX':'mean', 'TMIN':'mean' }).reset_index()
weather_impute.head(2)


# In[48]:


weather.head()


# In[49]:


weather_impute


# Merge the mean data with the weather data.

# In[50]:


weather = pd.merge(weather, weather_impute,  how='left', left_on=['MONTH','STATION'], right_on = ['MONTH','STATION']).rename(columns = {'TAVG_y':'TAVG_AVG',
                   'TMAX_y':'TMAX_AVG', 
                   'TMIN_y':'TMIN_AVG',
                   'TAVG_x':'TAVG',
                   'TMAX_x':'TMAX', 
                   'TMIN_x':'TMIN'})


# Check for missing values again.

# In[51]:


weather.TAVG[TAVG_idx] = weather.TAVG_AVG[TAVG_idx]
weather.TMAX[TMAX_idx] = weather.TMAX_AVG[TMAX_idx]
weather.TMIN[TMIN_idx] = weather.TMIN_AVG[TMIN_idx]
weather.isna().sum()


# Drop `STATION,MONTH,TAVG_AVG,TMAX_AVG,TMIN_AVG,TMAX,TMIN,SNWD` from the dataset.

# In[52]:


weather.drop(columns=['STATION','MONTH','TAVG_AVG', 'TMAX_AVG', 'TMIN_AVG', 'TMAX' ,'TMIN', 'SNWD'],inplace=True)


# Add the origin and destination weather conditions to the dataset.

# In[53]:


### Add origin weather conditions
data_orig = pd.merge(data_orig, weather,  how='left', left_on=['FlightDate','Origin'], right_on = ['DATE','airport']).rename(columns = {'AWND':'AWND_O','PRCP':'PRCP_O', 'TAVG':'TAVG_O', 'SNOW': 'SNOW_O'}).drop(columns=['DATE','airport'])

### Add destination weather conditions
data_orig = pd.merge(data_orig, weather,  how='left', left_on=['FlightDate','Dest'], right_on = ['DATE','airport']).rename(columns = {'AWND':'AWND_D','PRCP':'PRCP_D', 'TAVG':'TAVG_D', 'SNOW': 'SNOW_D'}).drop(columns=['DATE','airport'])


# **Note**: It is always a good practice to check nulls/NAs after joins.

# In[54]:


sum(data.isna().any())


# In[55]:


data_orig.columns


# Convert the categorical data into numerical data using one-hot encoding.

# In[56]:


data = data_orig.copy()
data = data[['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest','Distance','DepHourofDay', 'is_delay', 'is_holiday', 'AWND_O', 'PRCP_O',
       'TAVG_O', 'AWND_D', 'PRCP_D', 'TAVG_D', 'SNOW_O', 'SNOW_D']]


categorical_columns  = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 
       'Reporting_Airline', 'Origin', 'Dest', 'is_holiday','is_delay']
for c in categorical_columns:
    data[c] = data[c].astype('category')


# In[57]:


data_dummies = pd.get_dummies(data[['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'is_holiday']], drop_first=True)
data = pd.concat([data, data_dummies], axis = 1)
categorical_columns.remove('is_delay')
data.drop(categorical_columns,axis=1, inplace=True)


# Check the new columns.

# In[58]:


data.columns


# Rename the `is_delay` column to `target` again. Use the same code as before.

# In[59]:


data.rename(columns = {'is_delay':'target'}, inplace=True )


# Create the training sets again.

# In[60]:


train_features, test_features, train_labels, test_labels, val_features, val_labels = create_training_sets(data)

print(f"Length of train_features is: {train_features.shape}")
print(f"Length of train_labels is: {train_labels.shape}")
print(f"Length of val_features is: {val_features.shape}")
print(f"Length of val_labels is: {val_labels.shape}")
print(f"Length of test_features is: {test_features.shape}")
print(f"Length of test_labels is: {test_labels.shape}")


# ### New baseline classifier
# 
# Now, see if these new features add any predictive power to the model.

# In[61]:


# Instantiate the LinearLearner estimator object
num_classes = len(pd.unique(train_labels)) 
classifier_estimator = sagemaker.LinearLearner(role=sagemaker.get_execution_role(),
                                               instance_count=1,
                                               instance_type='ml.m4.xlarge',
                                               predictor_type='binary_classifier',
                                              binary_classifier_model_selection_criteria = 'cross_entropy_loss')


# In[62]:


train_records = classifier_estimator.record_set(train_features, train_labels, channel='train')
val_records = classifier_estimator.record_set(val_features, val_labels, channel='validation')
test_records = classifier_estimator.record_set(test_features, test_labels, channel='test')

classifier_estimator.fit([train_records, val_records, test_records])


# In[63]:


classifier_predictor = classifier_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# In[64]:


predict_batches(classifier_predictor, test_features, test_labels)


# The linear model shows only a little improvement in performance. Try a tree-based ensemble model called XGBoost with Amazon SageMaker.

# ### Try the XGBoost model

# Here are the steps you need to take:  
# 
# 1. Use the training set variables and save them as CSV files: train.csv, validation.csv and test.csv  
# 2. Store the bucket name in the variable. The Amazon S3 bucket name is provided to the left of the lab instructions.  
# a. `bucket = <LabBucketName>`  
# b. `prefix = 'sagemaker/xgboost'`  
# 3. Use Boto3 to upload the model to the bucket.    

# In[65]:


train_data, validation_data, test_data = np.split(data.sample(frac=1, random_state=1729), [int(0.7 * len(data)), int(0.9*len(data))])  

pd.concat([train_data['target'], train_data.drop(['target'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
pd.concat([validation_data['target'], validation_data.drop(['target'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)
pd.concat([test_data['target'], test_data.drop(['target'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)


# Replace **`<LabBucketName>`** with the resource name that was provided with your lab account.

# In[66]:


bucket = 'ml-pipeline-bucket'
prefix = 'sagemaker/xgboost'


# In[67]:


### Upload data to S3
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')


# In[68]:


from sagemaker.amazon.amazon_estimator import image_uris
container = image_uris.retrieve('xgboost',boto3.Session().region_name, '0.90-1')


# Use the `sagemaker.inputs.TrainingInput` function to create a record_set for the training and validation datasets.

# In[69]:


s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')


# In[70]:


sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role = sagemaker.get_execution_role(), 
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        eval_metric = "auc", 
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})


# Deploy the predictor for your new model and evaluate the model on the test dataset.

# In[71]:


from sagemaker.serializers import CSVSerializer

xgb_predictor = xgb.deploy(initial_instance_count = 1,
                           instance_type = 'ml.m4.xlarge', 
                           serializer=CSVSerializer())


# In[72]:


test_data = pd.concat([test_data['target'], test_data.drop(['target'], axis=1)], axis=1)


# In[73]:


def predict(predictor , features, labels , prob_threshold = 0.5, rows=500):
    """
    Return evaluation results
    predictor : Predictor object of model
    features: Input features to model
    label: Ground truth target values
    prob_threshold : Probability cut off to separate positive and negative class
    """
    split_array = np.array_split(features, int(features.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, predictor.predict(array).decode('utf-8')])

    preds = np.fromstring(predictions[1:], sep=',')
    preds = preds.reshape((-1,))
    predictions = np.where(preds > prob_threshold , 1, 0)
    labels = labels.reshape((-1,))
    
    
    # Calculate accuracy
    accuracy = (predictions == labels).sum() / labels.shape[0]
    print(f'Accuracy: {accuracy}')
    
    auc = roc_auc_score(labels, preds)
    print(f'AUC     : {auc}')
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average = 'binary')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1_score: {f1_score}')
    
    
    confusion_matrix = pd.crosstab(index=labels, columns=np.round(preds), rownames=['True'], colnames=['predictions']).astype(int)
    plt.figure(figsize = (5,5))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap="YlGnBu").set_title('Confusion Matrix') 
    return list(preds)


# In[74]:


predictions = predict(xgb_predictor, test_data.values[:, 1:] ,  test_data.iloc[:, 0:1].values, prob_threshold = 0.5)


# ### Try different thresholds

# In[75]:


predictions = predict(xgb_predictor, test_data.values[:, 1:] ,  test_data.iloc[:, 0:1].values, prob_threshold = 0.8)


# **Question**: What can you conclude from how the model did on the test set?

# **Answer:** By changing the threshold for positive class, we see that:
# - AUC doesnt change.  
# - Precision, recall, and accuracy change. We could use a threshold based on precision-recall trade-off or use metrics like AUC to gauge model performance.  

# ### Hyperparameter optimization (HPO)

# In[76]:


from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

### You can spin up multiple instances to do hyperparameter optimization in parallel

xgb = sagemaker.estimator.Estimator(container,
                                    role=sagemaker.get_execution_role(), 
                                    instance_count= 2, # make sure you have limit set for these instances
                                    instance_type='ml.m4.xlarge', 
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)

xgb.set_hyperparameters(eval_metric='auc',
                        objective='binary:logistic',
                        num_round=100,
                        rate_drop=0.3,
                        tweedie_variance_power=1.4)


# In[77]:


hyperparameter_ranges = {'eta': ContinuousParameter(0, 0.5),
                        'min_child_weight': ContinuousParameter(3, 10),
                         'num_round': IntegerParameter(10, 150),
                        'alpha': ContinuousParameter(0, 2),
                        'max_depth': IntegerParameter(10, 15)}

objective_metric_name = 'validation:auc'

tuner = HyperparameterTuner(xgb,
                            objective_metric_name,
                            hyperparameter_ranges,
                            max_jobs=10,
                            max_parallel_jobs=3)


# In[78]:


tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})


# In[79]:


boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']


# <i class="fas fa-exclamation-triangle" style="color:red"></i> Wait until the training job is finished. It may take 25-30 minutes.
# 
# **To monitor hyperparameter optimization jobs:**  
# 
# 1. In the AWS Management Console, on the **Services** menu, click **Amazon SageMaker**.  
# 1. Click **Training** > **Hyperparameter tuning jobs**.  
# 1. You can check the status of each hyperparametertuning job, its objective metric value, and its logs.  

# In[80]:


sage_client = boto3.Session().client('sagemaker')
tuning_job_name = tuner.latest_tuning_job.job_name

# Run this cell to check current status of hyperparameter tuning job
tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

status = tuning_job_result['HyperParameterTuningJobStatus']
if status != 'Completed':
    print('Reminder: the tuning job has not been completed.')
    
job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
print("%d training jobs have completed" % job_count)
    
is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']


from pprint import pprint
if tuning_job_result.get('BestTrainingJob',None):
    print("Best model found so far:")
    pprint(tuning_job_result['BestTrainingJob'])
else:
    print("No training jobs have reported results yet.")
    
### Get the hyperparameter tuning job results in a dataframe
tuner_df = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name).dataframe()

if len(tuner_df) > 0:
    df = tuner_df[tuner_df['FinalObjectiveValue'] > -float('inf')]
    if len(df) > 0:
        df = df.sort_values('FinalObjectiveValue', ascending=is_minimize)
        print("Number of training jobs with valid objective: %d" % len(df))
        print({"lowest":min(df['FinalObjectiveValue']),"highest": max(df['FinalObjectiveValue'])})
        pd.set_option('display.max_colwidth', -1)  # Don't truncate TrainingJobName        
    else:
        print("No training jobs have reported valid results yet.")
        
tuner_df


# Deploy the best model from the hyperparameter optimization training.

# In[81]:


xgb_predictor_hpo = tuner.deploy(initial_instance_count = 1,
                                 instance_type = 'ml.m4.xlarge',
                                serializer=CSVSerializer())


# In[82]:


predictions = predict(xgb_predictor_hpo, test_data.values[:, 1:] ,  test_data.iloc[:, 0:1].values, prob_threshold = 0.5)


# ### Feature importance
# 
# The model files for all hyperparameter tuning jobs are saved by the training job name as in the table above in the folder: "{bucket}/{prefix}/output/". You can load the model and use it like a regular sklearn model object for interpretation.

# In[86]:


# best_hpo_model_path = "s3://" + bucket + "/sagemaker/xgboost/output/<Best model TrainingJobName>/output/model.tar.gz"
best_hpo_model_path = "s3://" + bucket + "/sagemaker/xgboost/output/sagemaker-xgboost-201112-1435-009-7fafc153/output/model.tar.gz"
### Download best_hpo_model_path to local
get_ipython().system('aws s3 cp {best_hpo_model_path} .')


# In[84]:


get_ipython().system("pip install xgboost=='0.90'")


# #### Load picked model file

# In[87]:


import pickle
import tarfile
import xgboost

with open('model.tar.gz', 'rb') as f:
    with tarfile.open(fileobj=f, mode='r') as tar_f:
        with tar_f.extractfile('xgboost-model') as extracted_f:
            xgbooster = pickle.load(extracted_f)


# #### Map column names to XGBoost model

# In[88]:


columns = list(data.columns)
columns.remove('target')


# In[89]:


feature_importance = xgbooster.get_fscore()
feature_importance_col = {}

for column, fname in zip(columns, xgbooster.feature_names):
    try:
         feature_importance_col[column] = feature_importance[fname]
    except Exception:
        pass


# #### Sort by feature importance value

# In[90]:


sorted(feature_importance_col.items(), key=lambda kv: kv[1], reverse=True)


# Based on the feature importance above, you can see that DepHourofDay, Airwind, and Temperature at both source and destination are the major influencers in deciding the flight delay.
# 
# This is one way to validate what is intuitive in terms of features and what the model actually learns.

# ## Conclusion
# 
# You've now gone through at least a couple iterations of training and evaluating your model. It's time to wrap up this project and reflect on what you've learned and what types of steps you might take moving forward (assuming you had more time). Use the cell below to answer some of these and other relevant questions:
# 
# 1. Does your model performance meet your business goal? If not, what are some things you'd like to do differently if you had more time for tuning?
# 2. To what extent did your model improve as you made changes to your dataset, features, and hyperparameters? What types of techniques did you employ throughout this project that you felt yielded the greatest improvements in your model?
# 3. What were some of the biggest challenges you encountered throughout this project?
# 4. Are there outstanding questions you have about aspects of the pipeline that just didn't make sense to you?
# 5. What were the three most important things you learned about machine learning while completing this project?
# 
# #### <span style="color: blue;">Project presentation: Summarize your answers to these questions in your project presentation as well. Pull together all of your notes for your project presentation at this point and prepare to present your findings to the class.</span>

# In[ ]:




