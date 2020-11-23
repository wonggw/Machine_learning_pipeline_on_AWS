#!/usr/bin/env python
# coding: utf-8

# # Targeting Direct Marketing
# 
# ---
# 
# ## Background
# Direct marketing, either through mail, email, phone, etc., is a common tactic to acquire customers. Because resources and a customer's attention are limited, the goal is to only target the subset of prospects who are likely to engage with a specific offer. Predicting those potential customers based on readily available information like demographics, past interactions, and environmental factors is a common machine learning problem.
# 
# This notebook presents an example problem to predict if a customer will enroll for a term deposit at a bank after one or more phone calls. The steps include:
# 
# * Preparing your Amazon SageMaker notebook
# * Downloading data from the internet into Amazon SageMaker
# * Investigating and transforming the data so that it can be fed to Amazon SageMaker algorithms
# * Estimating a model 
# * Evaluating the effectiveness of the model
# * Setting the model up to make ongoing predictions
# 
# ---
# 
# ## Preparation
# 
# _This notebook was created and tested on an ml.m4.xlarge notebook instance._
# 
# Start by specifying:
# 
# - The Amazon Simple Storage Service (Amazon S3) bucket and prefix that you want to use for training and model data. This should be within the same Region as the Notebook Instance, training, and hosting.
# - The AWS Identity and Access Management (IAM) role Amazon Resource Name (ARN) used to give training and hosting access to your data. See the documentation for how to create these. 
# 
# **Note:** If more than one role is required for notebook instances, training, and/or hosting, please replace the boto regexp with the appropriate full IAM role ARN string(s).
# 
# Replace **`<LabBucketName>`** with the resource name that was provided with your lab account.

# In[1]:


#bucket = 'ml-pipeline-bucket'
bucket = 'ml-pipeline-bucket'
prefix = 'sagemaker/DEMO-xgboost-dm'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# Bring in the Python libraries that you'll use throughout the analysis.

# In[2]:


import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
import matplotlib.pyplot as plt                   # For charts and visualizations
from IPython.display import Image                 # For displaying images in the notebook
from IPython.display import display               # For displaying outputs in the notebook
from time import gmtime, strftime                 # For labeling Amazon SageMaker models, endpoints, etc.
import sys                                        # For writing outputs to the notebook
import math                                       # For ceiling function
import json                                       # For parsing hosting outputs
import os                                         # For manipulating filepath names
import sagemaker                                  # Amazon SageMaker's Python SDK provides many helper functions
from sagemaker.predictor import CSVSerializer    # Converts strings for HTTP POST requests on inference


# ---
# 
# ## Data
# Start by downloading the [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing) from the University of California, Irvine (UCI) Machine Learning ML Repository.
# 
# **Source information:**
# 
# S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

# In[3]:


get_ipython().system('aws s3 cp s3://aws-tc-largeobjects/ILT-TF-200-MLDWTS/lab1/bank-additional.zip .')
#!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
get_ipython().system('unzip -o bank-additional.zip')


# Now, read this into a Pandas dataframe and take a look.

# In[4]:


data = pd.read_csv('./bank-additional/bank-additional-full.csv', sep=';')
pd.set_option('display.max_columns', 500)     # Makes sure you can see all of the columns
pd.set_option('display.max_rows', 20)         # Keeps the output on one page


# In[5]:


data.head()


# Let's talk about the data.  At a high level, you can see:
# 
# * There are just over 40K customer records and 20 features for each customer.
# * The features are mixed--some numeric, some categorical.
# * The data appears to be sorted, at least by `time` and `contact`.
# 

# ### Exploration
# Let's explore the data. First, let's understand how the features are distributed.

# In[6]:


# Frequency tables for each categorical feature
for column in data.select_dtypes(include=['object']).columns:
    display(pd.crosstab(index=data[column], columns='% observations', normalize='columns')*100)


# In[7]:


# Histograms for each numeric feature
display(data.describe())
display(data.describe(include=np.object))
#%matplotlib inline
hist = data.hist(bins=30, sharey=True, figsize=(10, 10))


# Notice that:
# 
# - Almost 90% of the values for our target variable y are "no", so most customers did not subscribe to a term deposit.
# - Many of the predictive features take on values of "unknown". Some are more common than others. We should think carefully as to what causes a value of "unknown" (are these customers non-representative in some way?) and how to handle that.
# - Even if "unknown" is included as its own distinct category, what does it mean, given that those observations likely fall within one of the other categories of that feature?
# - Many of the predictive features have categories with few observations in them. If we find a small category to be highly predictive of our target outcome, do we have enough evidence to make a generalization about that?
# - Contact timing is particularly skewed--almost a third in May and less than 1% in December. What does this mean for predicting our target variable next December?
# - There are no missing values in our numeric features, or missing values have already been imputed.
#   - `pdays` takes a value near 1,000 for almost all customers. This is likely a placeholder value signifying no previous contact.
# - Several numeric features have a long tail. Do we need to handle these few observations with extremely large values differently?
# - Several numeric features (particularly the macroeconomic ones) occur in distinct buckets. Should these be treated as categorical?
# 
# Next, let's look at how our features relate to the target that we are attempting to predict.

# In[8]:


for column in data.select_dtypes(include=['object']).columns:
    if column != 'y':
        display(pd.crosstab(index=data[column], columns=data['y'], normalize='columns'))

for column in data.select_dtypes(exclude=['object']).columns:
    print(column)
    hist = data[[column, 'y']].hist(by='y', bins=30)
    plt.show()


# Notice that:
# - Customers who are "blue-collar", "married", "unknown" default status, contacted by "telephone", and/or in "May" are a substantially lower portion of "yes" than "no" for subscribing.
# - Distributions for numeric variables are different across "yes" and "no" subscribing groups, but the relationships may not be straightforward or obvious.
# 
# Now let's look at how our features relate to one another.

# In[9]:


display(data.corr())
pd.plotting.scatter_matrix(data, figsize=(12, 12))
plt.show()


# Notice that:
# 
# - Features vary widely in their relationships with one another. Some have highly negative correlation; others have highly positive correlation.
# - Relationships between features are non-linear and discrete in many cases.

# ### Transformation
# 
# Cleaning up data is part of nearly every machine learning project.  It arguably presents the biggest risk if done incorrectly and is one of the more subjective aspects in the process. 

# In[10]:


data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)                                 # Indicator variable to capture when pdays takes a value of 999
data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)   # Indicator for individuals not actively employed
model_data = pd.get_dummies(data)                                                                  # Convert categorical variables to sets of indicators


# In[11]:


model_data = model_data.drop(['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)


# ### Train-test split

# In[12]:


train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])   # Randomly sort the data; then split out first 70%, second 20%, and last 10%


# Amazon SageMaker's XGBoost container expects data in the libSVM or CSV data format. For this example, you'll stick to CSV. Note that the first column must be the target variable, and the CSV should not include headers. Also, notice that although repetitive, it's easiest to do this after the train|validation|test split rather than before. This avoids any misalignment issues due to random reordering.

# In[13]:


pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
pd.concat([validation_data['y_yes'], validation_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)


# Now, copy the file to Amazon S3 for Amazon SageMaker's managed training to pick up.

# In[14]:


boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')


# ---
# 
# ## Training
# 
# First, you need to specify the Amazon Elastic Container Registry (Amazon ECR) container location for Amazon SageMaker's implementation of XGBoost.

# In[15]:


container = sagemaker.image_uris.retrieve('xgboost',boto3.Session().region_name, '1.0-1')


# Then, because you're training with the CSV file format, create `s3_input`s for the training function to use as a pointer to the files in Amazon S3. This also specifies that the content type is CSV.

# In[16]:


s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')


# Now, you need to specify training parameters to the estimator. This includes:
# - The `xgboost` algorithm container   
# - The IAM role to use   
# - Training instance type and count   
# - Amazon S3 location for output data
# - Algorithm hyperparameters
# 
# And then a `.fit()` function, which specifies the Amazon S3 location for output data. In this case, both a training and validation set are passed in.

# In[17]:


sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
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
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 


# ---
# 
# ## Hosting
# Now that you've trained the `xgboost` algorithm on the data, deploy a model that's hosted behind a real-time endpoint.
# 
# First, you need to determine how to pass data into and receive data from the endpoint. The data is currently stored as NumPy arrays in the memory of your notebook instance. To send it in an HTTP POST request, you'll serialize it as a CSV string and then decode the resulting CSV.
# 
# **Note:** For inference with the CSV format, Amazon SageMaker XGBoost requires that the data does NOT include the target variable.

# In[18]:


xgb_predictor = xgb.deploy(initial_instance_count=1,
                           serializer=CSVSerializer(),
                           instance_type='ml.m4.xlarge')


# ---
# 
# ## Evaluation
# There are many ways to compare the performance of a machine learning model, but start by simply comparing actual to predicted values. In this case, you're simply predicting whether the customer subscribed to a term deposit (`1`) or not (`0`), which produces a simple confusion matrix.
# 

# Now, use a simple function to:  
# 
# 1. Loop over the test dataset  
# 1. Split it into mini-batches of rows   
# 1. Convert those mini-batches to CSV string payloads (notice that the target variable is dropped from the dataset first)  
# 1. Retrieve mini-batch predictions by invoking the XGBoost endpoint  
# 1. Collect predictions and convert from the CSV output that the model provides into a NumPy array  

# In[19]:


def predict(data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:], sep=',')

predictions = predict(test_data.drop(['y_no', 'y_yes'], axis=1).values)


# Now, check the confusion matrix to see how well the model predicted vs. actuals.

# In[20]:


pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions), rownames=['actuals'], colnames=['predictions'])


# Of about 4,000 potential customers, the model predicted that 136 would subscribe, and 94 actually did. There were also 389 customers who subscribed that were not predicted to subscribe. This is less than desirable, but the model can (and should) be tuned to improve this. Most importantly, note that with minimal effort, the model produced accuracies similar to those published [here](http://media.salford-systems.com/video/tutorial/2015/targeted_marketing.pdf).
# 
# **Note:** Because there is some element of randomness in the algorithm's subsample, your results may differ slightly from the text written above.

# ### Cleanup
# 
# If you are finished with this notebook, run the following cell. This removes the hosted endpoint you created and avoids any charges from a stray instance being left on.

# In[21]:


sagemaker.Session().delete_endpoint(xgb_predictor.endpoint_name)


# In[ ]:




