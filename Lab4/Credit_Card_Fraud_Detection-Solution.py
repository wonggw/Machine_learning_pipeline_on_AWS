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


# # Project Solution Lab 4: 

# # Iteration II

# # Step 4: Feature engineering
# 
# You've now gone through one iteration of training and evaluating your model. Given that the outcome you reached for your model the first time probably wasn't sufficient for solving your business problem, what are some things you could change about your data to possibly improve model performance?
# 
# ### Key questions to consider:
# 1. How might the balance of your two main classes (fraud and not fraud) impact model performance?
# 2. Does balancing your dataset have any impact on correlations between your features?
# 3. Are there feature reduction techniques you could perform at this stage that might have a positive impact on model performance? 
# 4. After performing some feature engineering, how does your model performance compare to the first iteration?
# 
# Use the cells below to perform specific feature engineering techniques (per the questions above) that you think could improve your model performance. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Record key decisions and methods you use in this section in your project presentations, as well as any new performance metrics you obtain after evaluating your model again.</span>

# Before you start, think about why the precision and recall are around 80% while the accuracy is 99%.

# **Answer**: Imbalanced datasets!

# The accuracy is calculated with how many examples the model got right. However, most of the examples are actually negative, so if you actually predict all examples as zero in this very imbalanced dataset, you can still get an accuracy of about 99.827%. Having an imbalanced dataset may cause some problems with algorithm performance. So it's useful to treat the imbalance in the data before you train the model.
# 
# **Question**: How do you solve the problem of dataset imbalance?

# **Answer:** You resample the dataset to create at least a balanced class distribution.

# **Question**: Print the shape of your datasets again.

# In[32]:


print(f"Length of train_features is: {train_features.shape}")
print(f"Length of train_labels is: {train_labels.shape}")
print(f"Length of val_features is: {val_features.shape}")
print(f"Length of val_labels is: {val_labels.shape}")
print(f"Length of test_features is: {test_features.shape}")
print(f"Length of test_labels is: {test_labels.shape}")


# Use `sns.countplot` to plot the original distribution of the dataset.

# In[33]:


sns.countplot(df['Class'])
plt.title('Original Distribution of the dataset')


# Convert `train_features` back into a DataFrame.

# In[34]:


df_train = pd.DataFrame(train_features, columns = df.columns.drop(['Time','Class']))
df_train['Target'] = train_labels


# In[35]:


df_train.head()


# There are two main ways to handle imbalanced datasets:
# 
# - Oversample to add more positive samples
#     - Random oversampling
#     - [Synthetic minority oversampling technique (SMOTE)](https://arxiv.org/abs/1106.1813)
# - Undersample to reduce the negative samples
#     - Random undersampling
#     - Generate centroids using clustering methods
# 
# You can use a library called `Imbalanced-learn` for sampling the datasets. `imbalanced-learn` is a Python package offering a number of resampling techniques commonly used in datasets showing strong between-class imbalance. It is compatible with scikit-learn and is part of scikit-learn-contrib projects. For more information, see [imbalanced-learn API documentation](https://imbalanced-learn.org/stable/introduction.html). 
# 
# Choose undersampling for this example first. To create the balanced dataset:
# 
# 1. Create a new DataFrame `fraud_df` with all the positive examples.
# 2. Create another DataFrame `non_fraud_df` and use `dataframe.sample` with the same number as the `fraud_df` DataFrame and `random_state=235`.
# 3. Concatenate both DataFrames into a new DataFrame `balanced_df`.

# In[36]:


fraud_df = df_train[df_train['Target'] == 1]

non_fraud_df = df_train[df_train['Target'] == 0].sample(n=len(fraud_df), random_state=235)

balanced_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True, sort=False)

balanced_df.head()


# Check the distribution and shape again using `sns.countplot()`.

# In[37]:


sns.countplot(balanced_df['Target'])
plt.title('Original Distribution of the dataset')


# In[38]:


balanced_df.shape


# Before looking at the training, look at what will happen if you use a feature reduction technique like t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

# In[39]:


from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(balanced_df.drop(columns = "Target"))
X_embedded.shape


# In[40]:


from matplotlib.colors import ListedColormap
plt.figure(figsize = (10,10))
plt.scatter(X_embedded[:,0], X_embedded[:,1], 
            c = balanced_df['Target'],
            s = 1,
            cmap = ListedColormap(['Red', 'Blue']),
            linewidths=1)

plt.title('Red: 0 , Blue: 1')


# **Question:** Does t-SNE help you differentiate the fraud from not fraud?
#     
# **Answer:** t-SNE can differentiate between some of the fraud and not fraud cases, but it doesn't do a good job of completely differentiating the two when we reduce the data to just two dimensions.

# Now that you have the new data, compare what the correlation matrix looks like before and after.

# In[41]:


# Make sure to use the subsample in the correlation

plt.figure(figsize = (20,10))
correlation_matrix_before = df.corr()
sns.heatmap(correlation_matrix_before, annot=True,fmt=".2f")

plt.figure(figsize = (20,10))
correlation_matrix_after = balanced_df.corr()
sns.heatmap(correlation_matrix_after, annot=True,fmt=".2f")


# **Question**: What can you deduce from looking at the different correlation matrices? If you see a difference, can you analyze why there is a difference?

# **Answer:** In the unbalanced set, the correlation matrix didn't show any **linear** relationships, but in this smaller balanced dataset, you can see some linear relationships. 
# 
# 

# **Question**: Would you drop any columns because of the correlated data?

# **Answer:** Yes, remove any features that have more than 90% correlation.

# Because there are some correlations, let's remove the correlated data that has more than 0.9 correlation. Run the following cell to drop the `V17` and `V18` columns.

# In[42]:


balanced_df_drop = balanced_df.drop(columns=['V17','V18'])


# Now it's time to train, deploy, and evaluate using the new balanced dataset.

# In[43]:


# Instantiate the LinearLearner estimator object
num_classes = len(pd.unique(train_labels))
linear_estimator_balanced = sagemaker.LinearLearner(role=sagemaker.get_execution_role(),
                                               instance_count=1,
                                               instance_type='ml.m4.xlarge',
                                               predictor_type='binary_classifier')


train_records_bal = linear_estimator_balanced.record_set(balanced_df.drop(['Target'], axis=1).values, 
                                                balanced_df['Target'].values, 
                                                channel='train')
val_records_bal = linear_estimator_balanced.record_set(val_features, val_labels, channel='validation')
test_records_bal = linear_estimator_balanced.record_set(test_features, test_labels, channel='test')

linear_estimator_balanced.fit([train_records_bal, val_records_bal, test_records_bal])


# Reducing the number of examples that make the distribution even caused the recall to go down rather than up. Let's try a different strategy, because we need a high recall.
# 
# Try using SMOTE to increase the number of positive examples.

# In[47]:


from imblearn.over_sampling import SMOTE 

X = df.drop(['Class', 'Time'], axis=1).values
y = df['Class'].values

sm = SMOTE(random_state=35)
X_res, y_res = sm.fit_resample(X, y)


# **Optional**: Convert the new dataset to a Pandas DataFrame and check the shape and distribution of the data.

# In[48]:


smote_df = pd.DataFrame(X_res,columns = df.drop(['Class', 'Time'], axis=1).columns) 
smote_df['Class'] = y_res
smote_df['Time'] = df['Time']


# Create new train, test, and validation datasets.

# In[49]:


#X_res, y_res = X_res.astype("float32"), y_res.astype("float32")

train_features, test_features, train_labels, test_labels, val_features, val_labels = create_training_sets(smote_df)


# Train your model using the new dataset.

# In[50]:


num_classes = len(pd.unique(train_labels))
linear_estimator_smote = sagemaker.LinearLearner(role=sagemaker.get_execution_role(),
                                               instance_count=1,
                                               instance_type='ml.m4.xlarge',
                                               predictor_type='binary_classifier')


train_records_smote = linear_estimator_smote.record_set(train_features, train_labels, channel='train')
val_records_smote = linear_estimator_smote.record_set(val_features, val_labels, channel='validation')
test_records_smote = linear_estimator_smote.record_set(test_features, test_labels, channel='test')

linear_estimator_smote.fit([train_records_smote, val_records_smote, test_records_smote])


# **Question**: What can you deduce from evaluating the training job?

# **Answer**: Test recall improved a lot by using SMOTE on the training dataset. We are looking at the recall value, which is at 93% now. This is great, but can we still improve this number by changing some parameters? Let's try to use hyperparameter optimization to see if it would improve.

# ### Hyperparameter optimization
# Another part of the model tuning phase is to perform hyperparameter optimization. This section gives you an opportunity to tune your hyperparameters to see the extent to which tuning improves your model performance. Use the following template code to help you get started launching an Amazon SageMaker hyperparameter tuning job and viewing the evaluation metrics. Use the following questions to help guide you through the rest of the section.
# 
# ### Key questions to consider:
# 1. How does the outcome of your objective metric of choice change as timing of your tuning job increases? What's the relationship between the different objective metrics you're getting and time? 
# 2. What is the correlation between your objective metric and the individual hyperparameters? Is there a hyperparameter that has a strong correlation with your objective metric? If so, what might you do to leverage this strong correlation?
# 3. Analyze the performance of your model after hyperparameter tuning. Is current performance sufficient for what you need to solve your business problem?
# 
# #### <span style="color: blue;">Project presentation: Record key decisions and methods you use in this section in your project presentations, as well as any new performance metrics you obtain after evaluating your model again.</span>

# In[51]:


from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

hyperparameter_ranges = {'wd': ContinuousParameter(1e-5, 1),
                        'l1': ContinuousParameter(1e-5, 0.01),
                        'learning_rate': ContinuousParameter(1e-4, 0.1)
                        }

objective_metric_name = 'test:recall'

tuner = HyperparameterTuner(linear_estimator_smote,
                            objective_metric_name,
                            hyperparameter_ranges,
                            max_jobs=10,
                            max_parallel_jobs=3)

tuner.fit([train_records_smote, val_records_smote, test_records_smote], include_cls_metadata=False)


# ### Track hyperparameter tuning job progress
# 
# After you launch a tuning job, you can see its progress by calling the `describe_tuning_job` API. The output from `describe-tuning-job` is a JSON object that contains information about the current state of the tuning job. You can call `list_training_jobs_for_tuning_job` to see a detailed list of the training jobs that the tuning job launched.

# In[52]:


client = boto3.Session().client('sagemaker')
tuning_job_result = client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)

status = tuning_job_result['HyperParameterTuningJobStatus']
while status != 'Completed':
    print('Reminder: the tuning job has not been completed.')
    
    job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
    print("%d training jobs have completed" % job_count)
    
    time.sleep(180)

    tuning_job_result = client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)
    status = tuning_job_result['HyperParameterTuningJobStatus']
    
print("\n\n All training jobs have completed")
is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']


# In[53]:


from pprint import pprint
if tuning_job_result.get('BestTrainingJob',None):
    print("Best model found so far:")
    pprint(tuning_job_result['BestTrainingJob'])
else:
    print("No training jobs have reported results yet.")


# ### Fetch all results as a DataFrame
# 
# You can list hyperparameters and objective metrics of all training jobs and pick up the training job with the best objective metric.

# In[54]:


tuner_analytics = sagemaker.HyperparameterTuningJobAnalytics(tuner.latest_tuning_job.job_name)

full_df = tuner_analytics.dataframe()

if len(full_df) > 0:
    df = full_df[full_df['FinalObjectiveValue'] > -float('inf')]
    if len(df) > 0:
        df = df.sort_values('FinalObjectiveValue', ascending=is_minimize)
        print("Number of training jobs with valid objective: %d" % len(df))
        print({"lowest":min(df['FinalObjectiveValue']),"highest": max(df['FinalObjectiveValue'])})
        pd.set_option('display.max_colwidth', -1)  # Don't truncate TrainingJobName        
    else:
        print("No training jobs have reported valid results yet.")
        
df


# **Question**: Does the model tuning help?

# **Answer:** The tuned model gives recall that is around the same value on the test set as the non-tuned model when it comes to getting the fraud data points right.

# ### See tuning job results vs. time
# 
# Next you will show how the objective metric changes over time, as the tuning job progresses. For Bayesian strategy, you should expect to see a general trend toward better results, but this progress will not be steady as the algorithm needs to balance exploration of new areas of parameter space against exploitation of known good areas. This can give you a sense of whether the number of training jobs is sufficient for the complexity of your search space.

# In[55]:


import bokeh
import bokeh.io
bokeh.io.output_notebook()
from bokeh.plotting import figure, show
from bokeh.models import HoverTool

class HoverHelper():

    def __init__(self, tuning_analytics):
        self.tuner = tuning_analytics

    def hovertool(self):
        tooltips = [
            ("FinalObjectiveValue", "@FinalObjectiveValue"),
            ("TrainingJobName", "@TrainingJobName"),
        ]
        for k in self.tuner.hyperparameter_ranges().keys():
            tooltips.append( (k, "@{%s}" % k) )

        ht = HoverTool(tooltips=tooltips)
        return ht

    def tools(self, standard_tools='pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset'):
        return [self.hovertool(), standard_tools]

hover = HoverHelper(tuner)

p = figure(plot_width=900, plot_height=400, tools=hover.tools(), x_axis_type='datetime')
p.circle(source=df, x='TrainingStartTime', y='FinalObjectiveValue')
show(p)


# ### Analyze the correlation between objective metric and individual hyperparameters
# 
# Now that you have finished a tuning job, you may want to know the correlation between your objective metric and individual hyperparameters you've selected to tune. Having that insight will help you decide whether it makes sense to adjust search ranges for certain hyperparameters and start another tuning job. For example, if you see a positive trend between the objective metric and a numerical hyperparameter, you probably want to set a higher tuning range for that hyperparameter in your next tuning job.
# 
# The following cell draws a graph for each hyperparameter to show its correlation with your objective metric.

# In[56]:


ranges = tuner_analytics.tuning_ranges
figures = []
for hp_name, hp_range in ranges.items():
    categorical_args = {}
    if hp_range.get('Values'):
        # This is marked as categorical.  Check if all options are actually numbers.
        def is_num(x):
            try:
                float(x)
                return 1
            except:
                return 0           
        vals = hp_range['Values']
        if sum([is_num(x) for x in vals]) == len(vals):
            # Bokeh has issues plotting a "categorical" range that's actually numeric, so plot as numeric
            print("Hyperparameter %s is tuned as categorical, but all values are numeric" % hp_name)
        else:
            # Set up extra options for plotting categoricals.  A bit tricky when they're actually numbers.
            categorical_args['x_range'] = vals

    # Now plot it
    p = figure(plot_width=600, plot_height=600, 
               title="Objective vs %s" % hp_name,
               tools=hover.tools(),
               x_axis_label=hp_name, y_axis_label=objective_name,
               **categorical_args)
    p.circle(source=df, x=hp_name, y='FinalObjectiveValue')
    figures.append(p)
show(bokeh.layouts.Column(*figures))


# Deploy this as your final model and evaluate it on the test set.

# In[57]:


tuned_model_deploy = tuner.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# In[58]:


predict_batches(tuned_model_deploy, test_features, test_labels)


# In[59]:


predict_batches(tuned_model_deploy, val_features, val_labels)


# ### OPTIONAL: Try the XGBoost algorithm
# Moving onto training, first we'll need to specify the locations of the XGBoost algorithm containers.

# In[60]:


containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}


bucket = sess.default_bucket()
prefix = 'sagemaker/xgboost-creditcard'

from sagemaker.amazon.amazon_estimator import image_uris
container = image_uris.retrieve('xgboost',boto3.Session().region_name, '1.0-1')


# Then, because you're training with the CSV file format, create s3_inputs that the training function can use as a pointer to the files in Amazon S3.

# In[61]:


train_features_balanced = balanced_df.drop(['Target'], axis=1).values
train_labels_balanced = balanced_df['Target'].values

train_features_label = np.insert(train_features_balanced, 0, train_labels_balanced, axis=1)
val_features_label = np.insert(val_features, 0, val_labels, axis=1)
test_features_label = np.insert(test_features, 0, test_labels, axis=1)

np.savetxt("train.csv", train_features_label, delimiter=",")
np.savetxt("validation.csv", val_features_label, delimiter=",")

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')

s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')


# Now, you can specify a few parameters like what type of training instances you'd like to use and how many, as well as the XGBoost hyperparameters. A few key hyperparameters are:
# 
# - `max_depth`: Controls how deep each tree within the algorithm can be built. Deeper trees can lead to better fit but are more computationally expensive and can lead to overfitting. There is typically some trade-off in model performance that needs to be explored between a large number of shallow trees and a smaller number of deeper trees.
# - `subsample`: Controls sampling of the training data. This technique can help reduce overfitting, but setting it too low can also starve the model of data.
# - `num_round`: Controls the number of boosting rounds. This is essentially the subsequent models that are trained using the residuals of previous iterations. Again, more rounds should produce a better fit on the training data but can be computationally expensive or lead to overfitting.
# - `eta`: Controls how aggressive each round of boosting is. Larger values lead to more conservative boosting.
# - `gamma`: Controls how aggressively trees are grown. Larger values lead to more conservative models.

# In[62]:


xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=10,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=1,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        eval_metric='auc',
                        num_round=100)


# First, you need to specify training parameters to the estimator. This includes:
# 
# - The XGBoost algorithm container
# - The IAM role to use
# - Training instance type and count
# - Amazon S3 location for output data
# - Algorithm hyperparameters
# 
# And then a `.fit()` function, which specifies the Amazon S3 location for output data. In this case, you have both a training and validation set that are passed in.

# In[63]:


xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})


# ### Hosting
# 
# Now that you've trained the XGBoost algorithm on the data, deploy a model that's hosted behind a real-time endpoint.

# Deploy your model on Amazon SageMaker.

# In[64]:


from sagemaker.predictor import CSVSerializer

xgb_predictor = xgb.deploy(initial_instance_count=1, 
                           instance_type='ml.m4.xlarge',
                          serializer=CSVSerializer())


# In[65]:


from sagemaker.predictor import csv_serializer 

def predict_xgboost(model, data, labels, rows=500):
    
    #model.content_type = 'text/csv'
    #model.serializer = csv_serializer
    #model.deserializer = None
    
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, model.predict(array).decode('utf-8')])
        
    preds = np.fromstring(predictions[1:], sep=',')
    confusion_matrix = pd.crosstab(index=labels, columns=np.round(preds), rownames=['True'], colnames=['predictions']).astype(int)
    plt.figure(figsize = (5,5))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap="YlGnBu").set_title('Confusion Matrix') 

predict_xgboost(xgb_predictor, test_features, test_labels)


# In[84]:


from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

hyperparameter_ranges_xgb = {'eta': ContinuousParameter(0.01, 0.2),
                         'max_depth': IntegerParameter(3, 9),
                         'gamma': IntegerParameter(0, 5),
                         'min_child_weight': IntegerParameter(2, 6),
                         'subsample': ContinuousParameter(0.5, 0.9),
                         'colsample_bytree': ContinuousParameter(0.5, 0.9)}

objective_metric_name_xgb = 'validation:auc'

tuner_xgb = HyperparameterTuner(xgb,
                            objective_metric_name_xgb,
                            hyperparameter_ranges_xgb,
                            max_jobs=10,
                            max_parallel_jobs=1)

tuner_xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})


# In[85]:


client = boto3.Session().client('sagemaker')
tuning_job_result = client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner_xgb.latest_tuning_job.job_name)

status = tuning_job_result['HyperParameterTuningJobStatus']
while status != 'Completed':
    print('Reminder: the tuning job has not been completed.')
    
    job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
    print("%d training jobs have completed" % job_count)
    
    time.sleep(180)

    tuning_job_result = client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner_xgb.latest_tuning_job.job_name)
    status = tuning_job_result['HyperParameterTuningJobStatus']
    
print("Training jobs have completed")
is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']


# Deploy your tuned model on Amazon SageMaker.

# In[86]:


xgb_predictor_tuned = tuner_xgb.deploy(initial_instance_count=1, 
                                       instance_type='ml.m4.xlarge',
                                       serializer=CSVSerializer() )


# In[87]:


predict_xgboost(xgb_predictor_tuned, test_features, test_labels)


# ## Conclusion
# 
# You've now gone through at least a couple iterations of training and evaluating your model. It's time to wrap up this project and reflect on what you've learned and what types of steps you might take moving forward (assuming you had more time). Use the cell below to answer some of these and other relevant questions:
# 
# 1. Does your model performance meet your business goal? If not, what are some things you'd do differently if you had more time for tuning?
# 2. To what extent did your model improve as you made changes to your dataset, features, and hyperparameters? What techniques did you employ throughout this project that you felt yielded the greatest improvements in your model?
# 3. What were some of the biggest challenges you encountered throughout this project?
# 4. What outstanding questions do you have about aspects of the pipeline that just didn't make sense to you?
# 5. What were the three most important things you learned about machine learning while completing this project?
# 
# #### <span style="color: blue;">Project presentation: Summarize your answers to these questions in your project presentation as well. Pull together all of your notes for your project presentation at this point and prepare to present your findings to the class.</span>

# In[ ]:




