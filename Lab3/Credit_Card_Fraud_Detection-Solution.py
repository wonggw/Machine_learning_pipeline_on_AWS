#!/usr/bin/env python
# coding: utf-8

# # Project Solution Lab 2: 

# # Problem: Predicting Credit Card Fraud 
# 
# ## Introduction to business scenario
# You work for a multinational bank. There has been a significant increase in the number of customers experiencing credit card fraud over the last few months. A major news outlet even recently published a story about the credit card fraud you and other banks are experiencing. 
# 
# As a response to this situation, you have been tasked to solve part of this problem by leveraging machine learning to identify fraudulent credit card transactions before they have a larger impact on your company. You have been given access to a dataset of past credit card transactions, which you can use to train a machine learning model to predict if transactions are fraudulent or not. 
# 
# 
# ## About this dataset
# The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred over the course of two days and includes examples of both fraudulent and legitimate transactions.
# 
# ### Features
# The dataset contains over 30 numerical features, most of which have undergone principal component analysis (PCA) transformations because of personal privacy issues with the data. The only features that have not been transformed with PCA are 'Time' and 'Amount'. The feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction amount. 'Class' is the response or target variable, and it takes a value of '1' in cases of fraud and '0' otherwise.
# 
# Features: 
# `V1, V2, ... V28`: Principal components obtained with PCA
# 
# Non-PCA features:
# - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset, $T_x - t_0$
# - `Amount`: Transaction amount; this feature can be used for example-dependent cost-sensitive learning 
# - `Class`: Target variable where `Fraud = 1` and `Not Fraud = 0`
# 
# ### Dataset attributions
# Website: https://www.openml.org/d/1597
# 
# Twitter: https://twitter.com/dalpozz/status/645542397569593344
# 
# Authors: Andrea Dal Pozzolo, Olivier Caelen, and Gianluca Bontempi
# Source: Credit card fraud detection - June 25, 2015
# Official citation: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
# 
# The dataset has been collected and analyzed during a research collaboration of Worldline and the Machine Learning Group (mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML.

# # Step 1: Problem formulation and data collection
# 
# Start this project off by writing a few sentences below that summarize the business problem and the business goal you're trying to achieve in this scenario. Include a business metric you would like your team to aspire toward. With that information defined, clearly write out the machine learning problem statement. Finally, add a comment or two about the type of machine learning this represents.
# 
# #### <span style="color: blue;">Project presentation: Include a summary of these details in your project presentations.</span>

# ### Read through a business scenario and:
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
# 

# ### Setup
# 
# Now that we have decided where to focus our energy, let's set things up so you can start working on solving the problem.
# 
# **Note:** This notebook was created and tested on an `ml.m4.xlarge` notebook instance. 

# In[1]:


# Import various Python libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Install imblearn
get_ipython().system('pip uninstall scikit-learn -y')
get_ipython().system('pip install imbalanced-learn==0.5.0')
get_ipython().system('pip install imblearn')


# ### Downloading the dataset

# In[3]:


# Check whether the file is already in the desired path or if it needs to be downloaded
import os
import subprocess
base_path = '/home/ec2-user/SageMaker/project/data/FraudDetection'
file_path = '/fraud.csv'

if not os.path.isfile(base_path + file_path):
    subprocess.run(['mkdir', '-p', base_path])
    subprocess.run(['aws', 's3', 'cp', 
                    's3://aws-tc-largeobjects/ILT-TF-200-MLDWTS/credit_card_project/', 
                    base_path,'--recursive'])
else:
    print('File already downloaded!')


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

# Read the CSV data into a Pandas dataframe. You can use the built-in Python `read_csv` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)).

# In[4]:


df = pd.read_csv('/home/ec2-user/SageMaker/project/data/FraudDetection/fraud.csv')


# Check the dataframe by printing the first 5 rows of the dataset.  
# 
# **Hint**: Use the `<dataframe>.head()` function.

# In[5]:


df.head()


# In[6]:


# The class has a weird string instead of a boolean or numbers 0 and 1, so convert it to 0 and 1 

mapped_class = {"'0'": 0, "'1'": 1}
df['Class'] = df['Class'].map(lambda x: mapped_class[x])


# In[7]:


# Check if that worked

df.head()


# **Question**: What is the shape of your dataset?  
# 
# **Hint**: To check the shape of a dataframe, use the `<dataframe>.shape` function.

# In[8]:


df.shape


# **Task**: Validate all the columns in the dataset and see that they are what you read above: `V1-V28`, `Time`, `Amount`, and `Class`.  
# 
# **Hint**: Use `<dataframe>.columns` to check the columns in your dataframe.

# In[9]:


df.columns


# **Question**: What can you find out about the column types and the null values? How many columns are numerical or categorical? 
# 
# **Hint**: Use the `info()` function to check.

# In[10]:


df.info()


# **Question**: Perform basic statistics using the Pandas library and `Describe` function. What is the mean and standard deviation for the `amount` feature? What can you deduce from those numbers?

# In[11]:


df.describe()


# **Answer:** Using the `describe` function, you can find some salient features of the dataset. 
# 
# - The mean amount for this dataset is \\$88 but this also consists of a lot of not fraud examples. But the max amount goes up to \\$25.
# - The standard deviation for the Amount column is \\$250.

# **Question**: What is the mean, standard deviation, and maximum for the `amount` for the records that are fraud?  
# 
# **Hint**: Use the built-in `mean()`, `std()`, and `max()` functions in dataframes.

# In[12]:


print("Fraud Statistics")

avg_amt = df[df['Class']== 1]['Amount'].mean()
std_dev = df[df['Class']== 1]['Amount'].std()
max_amt = df[df['Class']== 1]['Amount'].max()

print(f"The average amount is {avg_amt}")
print(f"The std deviation for amount is {std_dev}")
print(f"The max amount is {max_amt}")


# Now look at the target variable, `Class`. First, you can find out what the distribution is for it.
#  
# **Question**: What is the distribution of the classes?  
# 
# **Hint**: Use `<dataframe>['column_name'].value_counts()` to check the distribution.

# In[13]:


df['Class'].value_counts()


# **Question**: What can you deduce from the distribution of the classes?

# **Answer:** The target variable distribution is very skewed. There are a lot of **not fraud** examples and very few **fraud** examples. 

# **Question**: What's the ratio of classes for 0s to the total number of records?

# In[14]:


284315/(284315+492)


# ## Visualize your data
# If you haven't done so in the above, you can use the space below to further visualize some of your data. Look specifically at the distribution of features like `Amount` and `Time`, and also calculate the linear correlations between the features in the dataset. 
# 
# ### Specific questions to consider
# 1. After looking at the distributions of features like `Amount` and `Time`, to what extend might those features help your model? Is there anything you can deduce from those distributions that might be helpful in better understanding your data?
# 
# 2. Do the distributions of features like `Amount` and `Time` differ when you are looking only at data that is labeled as fraud?
# 
# 3. Are there any features in your dataset that are strongly correlated? If so, what would be your next steps?
# 
# Use the cells below to visualize your data and answer these and other questions that might be of interest to you. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Include a summary of your answers to these and similar questions in your project presentations.</span>

# Start with a simple scatter plot. Plot V1 vs. V2. For more information about plotting a scatter plot, see the [Matplotlib documentation](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html).

# In[15]:


plt.plot(df['V1'], df['V2'],'.')


# Look at a distribution of some of the features. Use `sns.distplot()` to find the distribution of the individual features such as `Amount` and `Time`.

# In[16]:


sns.distplot(df['Amount'])


# **Question**: Would the `Time` feature help you in any way? Look at the distribution again. What can you deduce from the scatter plot?

# **Answer:** This distribution plot tells us the same thing that we saw in the `describe` function above. Most of the values are closer to \\$100, but there are some that are bigger.

# Plot a histogram and Kernel density estimation (KDE).

# In[17]:


sns.distplot(df['Time'])


# The time distribution also confirms what the dataset told us. The data includes two days' worth of data where there are more credit card transactions during the day and fewer at night.

# **Question**: What does this distribution look like for fraud cases for the `Amount` column?

# In[18]:


sns.distplot(df[df['Class'] == 1]['Amount'])


# Now let's look at a distribution using a Seaborn function called `pairplot`. `pairplot` creates a grid of scatterplots, such that each feature in the dataset is used once as the X-axis and once as the Y-axis. The diagonal of this grid shows a distribution of the data for that feature. 
# 
# Look at `V1`, `V2`, `V2`, `V4`, and `Class` pairplots. What do you see in the plots? Can you differentiate the fraud and not fraud from these features?  
# 
# **Hint**: Create a new dataframe with columns `V1`, `V2`, `V4`, and `Class`.

# In[19]:


new_df = df[['V1', 'V2', 'V3', 'V4','Class']]
sns.pairplot(new_df, hue="Class")


# You can see for the smaller subset of the features that we used, there is a way to differentiate the fraud and not fraud, but it's not easy to separate it based on any one feature.
# 
# Now, let's look at how these features interact with each other. Use the Pandas `corr` function to calculate the linear correlations between the features of the dataset. It is always easier to visualize the correlation. Plot the correlation using the Seaborn heatmap (`sns.heatmap`) function with the `annot` flag set to `True`.

# In[20]:


plt.figure(figsize = (25,15))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True,fmt=".2f")


# **Question**: For correlated features, you should remove one of them before model training. Do you see any features that you can remove?

# **Answer**: There aren't any correlated features to be removed.

# # Project Solution Lab 3: 

# # Step 3: Model training and evaluation
# 
# There are some preliminary steps that you have to include when converting the dataset from a DataFrame to a format that a machine learning algorithm can use. For Amazon SageMaker, here are the steps you need to take:
# 
# 1. Split the data into `train_data`, `validation_data`, and `test_data` using `sklearn.model_selection.train_test_split`.    
# 2. Convert the dataset to an appropriate file format that the Amazon SageMaker training job can use. This can be either a CSV file or record protobuf. For more information, see [Common Data Formats for Training](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html).    
# 3. Upload the data to your Amazon S3 bucket.
# 
# Use the following cells to complete these steps. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Take note of the key decisions you've made in this phase in your project presentations.</span>

# - The Amazon Simple Storage Service (Amazon S3) bucket and prefix(?) that you want to use for training and model data. This should be within the same Region as the Notebook Instance, training, and hosting.
# - The AWS Identity and Access Management (IAM) role [Amazon Resource Name (ARN)](https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html) used to give training and hosting access to your data. See the documentation for how to create these.
# 
# **Note:** If more than one role is required for notebook instances, training, and/or hosting, replace the `get_execution_role()` call with the appropriate full IAM role ARN string(s).
# 
# Replace **`<LabBucketName>`** with the resource name that was provided with your lab account.

# In[21]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import image_uris
from sagemaker.amazon.amazon_estimator import RecordSet

# Instantiate an Amazon SageMaker session
sess = sagemaker.Session()

# Get the Amazon SageMaker role 
role = get_execution_role()

# Bucket name
bucket = 'ml-pipeline-bucket'

# Get the image URI for the container that includes the linear learner algorithm
container = image_uris.retrieve('linear-learner',boto3.Session().region_name)

print(f'Session {sess}')
print(f'The role is {role}')
print(f'The container is {role} in the {boto3.Session().region_name} region')


# In[22]:


import numpy as np
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
    ys = np.array(data["Class"]).astype("float32")
    
    # Drop all the unwanted columns including the target column
    drop_list = ["Class","Time"]
    
    # Drop the columns from the drop_list and convert the data into a NumPy array of type float32
    xs = np.array(data.drop(drop_list, axis=1)).astype("float32")
    
    np.random.seed(0)
    
    # Use the sklearn function train_test_split to split the dataset in the ratio train 80% and test 20%
    train_features, test_features,     train_labels, test_labels = train_test_split(xs, ys, test_size=0.2)
    
    # Use the sklearn function again to split the test dataset into 50% validation and 50% test
    val_features, test_features,     val_labels, test_labels = train_test_split(test_features, test_labels, test_size=0.5)
    
    return train_features, test_features, train_labels, test_labels, val_features, val_labels


# In[23]:


# Use the function to create your datasets
train_features, test_features, train_labels, test_labels, val_features, val_labels = create_training_sets(df)

print(f"Length of train_features is: {train_features.shape}")
print(f"Length of train_labels is: {train_labels.shape}")
print(f"Length of val_features is: {val_features.shape}")
print(f"Length of val_labels is: {val_labels.shape}")
print(f"Length of test_features is: {test_features.shape}")
print(f"Length of test_labels is: {test_labels.shape}")


# ### Model training
# 
# Lets start by instantiating the LinearLearner estimator with `predictor_type='binary_classifier'` parameter with one ml.m4.xlarge instance.

# In[24]:


import sagemaker
from sagemaker.amazon.amazon_estimator import RecordSet
import boto3

# Instantiate the LinearLearner estimator object
num_classes = len(pd.unique(train_labels))

# Instantiate the LinearLearner estimator 'binary classifier' object with one ml.m4.xlarge instance
linear = sagemaker.LinearLearner(role=sagemaker.get_execution_role(),
                                               instance_count=1,
                                               instance_type='ml.m4.xlarge',
                                               predictor_type='binary_classifier')


# Linear learner accepts training data in protobuf or CSV content types, and accepts inference requests in protobuf, CSV, or JSON content types. Training data has features and ground-truth labels, while the data in an inference request has only features. In a production pipeline, it is recommended to convert the data to the Amazon SageMaker protobuf format and store it in Amazon S3. However, to get up and running quickly, AWS provides the convenient method `record_set` for converting and uploading when the dataset is small enough to fit in local memory. It accepts NumPy arrays like the ones you already have, so let's use it here. The `RecordSet` object will keep track of the temporary Amazon S3 location of your data. Use the `estimator.record_set` function to create train, validation, and test records. Then, use the `estimator.fit` function to start your training job.

# In[25]:


### Create train, val, test records
train_records = linear.record_set(train_features, train_labels, channel='train')
val_records = linear.record_set(val_features, val_labels, channel='validation')
test_records = linear.record_set(test_features, test_labels, channel='test')


# Now, lets train your model on the dataset that you just uplaoded.

# In[26]:


### Fit the classifier
linear.fit([train_records,val_records,test_records], wait=True, logs='All')


# ## Model evaluation
# In this section, you'll evaluate your trained model. First, use the `estimator.deploy` function with `initial_instance_count= 1` and `instance_type= 'ml.m4.xlarge'` to deploy your model on Amazon SageMaker.

# In[27]:


linear_predictor = linear.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# Now that you have a hosted endpoint running, you can make real-time predictions from the model easily by making an http POST request. But first, you'll need to set up serializers and deserializers for passing your `test_features` NumPy arrays to the model behind the endpoint. You will also calculate the confusion matrix for your model to evaluate how it has done on your test data visually.

# In[28]:


from sklearn.metrics import accuracy_score,precision_score, recall_score
#from sagemaker.predictor import csv_serializer, json_deserializer,numpy_deserializer
#from sagemaker.predictor import csv_deserializer

def predict_batches(model, features, labels, split=200):
    """
    Predict datapoints in batches specified by the split. 
    The data will be split into <split> parts and model.predict is called 
    on each part
    Arguments:
        model: The model that you will use to call predict function
        features: The dataset to predict on
        labels: The true value of the records
        split: Number of parts to split the data into
    Returns:
        None
    """
    
    split_array = np.array_split(features, split)
    predictions = []
    for array in split_array:
        predictions +=  model.predict(array)

    # preds = np.array([p['predicted_label'] for p in predictions])
    preds = [i.label['predicted_label'].float32_tensor.values[0] for i in predictions]
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    print(f'Accuracy: {accuracy}')
    
    # Calculate precision
    precision = precision_score(labels, preds)
    print(f'Precision: {precision}')
    
    # Calculate recall
    recall = recall_score(labels, preds)
    print(f'Recall: {recall}')
    
    confusion_matrix = pd.crosstab(index=labels, columns=np.round(preds), rownames=['True'], colnames=['predictions']).astype(int)
    plt.figure(figsize = (5,5))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap="YlGnBu").set_title('Confusion Matrix') 
    


# Now that your endpoint is 'InService', evaluate how your model performs on the test set. Compare that test set performance to the performance on the training set. 
# 
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

# In[29]:


predict_batches(linear_predictor, test_features, test_labels)


# Similar to the test set, you can also look at the metrics for the training set. Keep in mind that those are also shown to you above in the logs.

# In[30]:


#predict_batches(linear_predictor, train_features, train_labels)


# In[31]:


# Delete inference endpoint
sagemaker.Session().delete_endpoint(linear_predictor.endpoint_name)

