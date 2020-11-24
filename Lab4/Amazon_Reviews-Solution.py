#!/usr/bin/env python
# coding: utf-8

# # Project Solution Lab 2: 

# # Problem: Recommend Movies or Shows to Users

# Modified from:
# - [Implementing a Recommender System with SageMaker, MXNet, and Gluon](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_applying_machine_learning/gluon_recommender_system/gluon_recommender_system.ipynb)
# - [An Introduction to Factorization Machines with MNIST](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/factorization_machines_mnist/factorization_machines_mnist.ipynb)
# - [Extending Amazon SageMaker Factorization Machines Algorithm to Predict Top X Recommendations](https://aws.amazon.com/blogs/machine-learning/extending-amazon-sagemaker-factorization-machines-algorithm-to-predict-top-x-recommendations/)

# ## Introduction to business scenario
# 
# You work for a startup that focuses on delivering on-demand video streaming services to users. The company wants to introduce movie/show recommendations for their users based on their viewing history.
# 
# You are tasked with solving part of this problem by leveraging machine learning to create a recommendation engine to be used on the user website. You are given access to the dataset of historical user preferences and the movies they watched. You can use this to train a machine learning model to recommend movies/shows to watch.
# 
# ## About this dataset  
# The Amazon Customer Reviews Dataset is a collection of reviews on different products from the Amazon.com marketplace from 1995 until 2015. Customer reviews are one of the most important data types at Amazon. Collecting and showing reviews has been part of the Amazon culture since the beginning of the company and is arguably one important source of innovation. For more details on this dataset, see [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html).
# 
# This exercise focuses on reviews of videos. The videos dataset contains 1- to 5-star ratings from over 2M Amazon customers on 160K digital videos.

# ### Features
# 
# **Data columns**
# 
# - `marketplace`: Two-letter country code (in this case, all "US")
# - `customer_id`: Random identifier that can be used to aggregate reviews written by a single author
# - `review_id`: Unique ID for the review
# - `product_id`: Amazon Standard Identification Number (ASIN). http://www.amazon.com/dp/<ASIN\> links to the product's detail page.
# - `product_parent`: The parent of that ASIN. Multiple ASINs (color or format variations of the same product) can roll up into a single parent.
# - `product_title`: Title description of the product
# - `product_category`: Broad product category that can be used to group reviews (in this case, digital videos)
# - `star_rating`: Product's rating (1 to 5 stars)
# - `helpful_votes`: Number of helpful votes for the review
# - `total_votes`: Number of total votes the review received
# - `vine`: Was the review written as part of the Vine program?
# - `verified_purchase`: Was the review from a verified purchase?
# - `review_headline`: Title of the review itself
# - `review_body`: Text of the review
# - `review_date`: Date the review was written 
# 
# **Data format**
# - Tab `\t` separated text file, without quote or escape characters
# - First line in each file is header; 1 line corresponds to 1 record
# 
# ### Dataset attributions
# 
# Website: https://s3.amazonaws.com/amazon-reviews-pds/readme.html
# 
# This dataset is being provided to you by permission of Amazon and is subject to the terms of the AWS Digital Training Service Agreement (available at https://aws.amazon.com/training/digital-training-agreement). You are expressly prohibited from copying, modifying, selling, exporting, or using this dataset in any way other than for the purpose of completing this lab.

# ## Brainstorming and designing a question...
# 
# ...That you can answer with machine learning. 
# 
# The first step in most projects is to think about the question you want to ask, how the data available supports this question, and which tool (in this case, machine learning model) you are going to use to answer the question. This is an important step because it helps narrow the scope of exploration and gives clarity on the features that you are going to use. 
# 
# Take a moment to write your thoughts regarding the dataset in the cell below. What are the things you can predict with machine learning? Why may that be relevant from a business/client perspective? Explain why you consider these thoughts important.

# In[1]:


# Write your thoughts here


# There might be several ideas about what to do with the data, but for now we are all going to work on recommending a video to a particular user.

# ## Recommendation and factorization machines
# 
# In many ways, recommender systems were a catalyst for the current popularity of machine learning. One of Amazon's earliest successes was the "Customers who bought this, also bought..." feature. The million dollar Netflix Prize spurred research, raised public awareness, and inspired numerous other data science competitions.
# 
# Recommender systems can utilize a multitude of data sources and machine learning algorithms. Most combine various unsupervised, supervised, and reinforcement learning techniques into a holistic framework. However, the core component is almost always a model that predicts a user's rating (or purchase) for a certain item based on that user's historical ratings of similar items as well as the behavior of other similar users. The minimal required dataset for this is a history of user item ratings (which we have).
# 
# The method that you'll use is a factorization machine. A factorization machine is a general-purpose supervised learning algorithm that you can use for both classification and regression tasks. It is an extension of a linear model and is designed to parsimoniously (simply) capture interactions between features in high-dimensional sparse datasets. This makes it a good candidate to handle data patterns with features such as click prediction and item recommendation.

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
# 
# Start by specifying:
# - The Amazon Simple Storage Service (Amazon S3) bucket and prefix(?) that you want to use for training and model data. This should be within the same Region as the Notebook Instance, training, and hosting.
# - The AWS Identity and Access Management (IAM) role [Amazon Resource Name (ARN)](https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html) used to give training and hosting access to your data. See the documentation for how to create these.
# 
# **Note:** If more than one role is required for notebook instances, training, and/or hosting, replace the `get_execution_role()` call with the appropriate full IAM role ARN string(s).

# Replace **`<LabBucketName>`** with the resource name that was provided with your lab account.

# In[2]:


# Change the bucket and prefix according to your information
#bucket = '<LabBucketName>'
bucket = 'ml-pipeline-bucket'
prefix = 'sagemaker-fm' 

import sagemaker
role = sagemaker.get_execution_role()


# Now, load some Python libraries you'll need for the remainder of this example notebook.

# In[3]:


import os, subprocess
import warnings
import pandas as pd
import numpy as np
import sagemaker
from sagemaker.mxnet import MXNet
import boto3
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add this to display all the outputs in the cell and not just the last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Ignore warnings
warnings.filterwarnings("ignore")


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
# 
# Start by bringing in the dataset from an Amazon S3 public bucket to this notebook environment.

# In[4]:


# Check whether the file is already in the desired path or if it needs to be downloaded

base_path = '/home/ec2-user/SageMaker/project/data/AmazonReviews'
file_path = '/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz'

if not os.path.isfile(base_path + file_path):
    subprocess.run(['mkdir', '-p', base_path])
    subprocess.run(['aws', 's3', 'cp', 's3://amazon-reviews-pds/tsv' + file_path, base_path])
else:
    print('File already downloaded!')


# ### Reading the dataset
# 
# Read the data into a Pandas dataframe so that you can know what you are dealing with.
# 
# **Note:** You'll set `error_bad_lines=False` when reading the file in, because there appear to be a very small number of records that would create a problem otherwise.
# 
# **Hint:** You can use the built-in Python `read_csv` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)). You can use the file path directly with Pandas `read_csv` with `delimiter='\t'`.
# 
# For example: `pd.read_csv('filename.tar.gz', delimiter = '\t', error_bad_lines=False)`

# In[5]:


df = pd.read_csv(base_path + file_path, 
                 delimiter='\t',
                 error_bad_lines=False)


# Print the first few rows of your dataset.
# 
# **Hint**: Use the `pandas.head(<number>)` function to print the rows.

# In[6]:


df.head(3)


# Now what is the information contained in all the columns?

# ### Anatomy of the dataset
# 
# Get a little more comfortable with the data and see what features are at hand.
# 
# - `marketplace`: Two-letter country code (in this case, all "US")
# - `customer_id`: Random identifier that can be used to aggregate reviews written by a single author
# - `review_id`: Unique ID for the review
# - `product_id`: Amazon Standard Identification Number (ASIN). http://www.amazon.com/dp/<ASIN\> links to the product's detail page.
# - `product_parent`: The parent of that ASIN. Multiple ASINs (color or format variations of the same product) can roll up into a single parent.
# - `product_title`: Title description of the product
# - `product_category`: Broad product category that can be used to group reviews (in this case, digital videos)
# - `star_rating`: Product's rating (1 to 5 stars)
# - `helpful_votes`: Number of helpful votes for the review
# - `total_votes`: Number of total votes the review received
# - `vine`: Was the review written as part of the Vine program?
# - `verified_purchase`: Was the review from a verified purchase?
# - `review_headline`: Title of the review itself
# - `review_body`: Text of the review
# - `review_date`: Date the review was written

# ### Analyzing and processing the dataset
# 
# #### Exploring the data

# **Question:** How many rows and columns do you have in the dataset?

# Check the size of the dataset.  
# 
# **Hint**: Use the `<dataframe>.shape` function to check the size of your dataframe

# In[7]:


df.shape


# Answer: (3998345,15)

# **Question:** Which columns contain null values, and how many null values do they contain?

# Print a summary of the dataset.
# 
# **Hint**: Use `<dataframe>.info` function using the keyword arguments `null_counts = True`

# In[8]:


df.info(null_counts=True)


# **Answer:** Review headline: 25, Review_body: 78, Review_date: 138

# **Question:** Are there any duplicate rows? If yes, how many are there?
# 
# **Hint**: Filter the dataframe using `dataframe.duplicated()` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html#pandas.DataFrame.duplicated)) and check the length of the new dataframe.

# In[9]:


duplicates = df[df.duplicated()]

len(duplicates)


# **Answer:** There are no duplicated rows.

# ### Data preprocessing
# 
# Now it's time to decide what features you are going to use and how you are going to prepare them for your model. For this example, limit yourself to `customer_id`, `product_id`, `product_title`, and `star_rating`. Including additional features in the recommendation system could be beneficial but would require substantial processing (particularly the text data), which would be beyond the scope of this notebook.
# 
# Reduce this dataset and only use the columns mentioned.
# 
# **Hint:** Select multiple columns as a dataframe by passing the columns as a list. For example: `df[['column_name 1', 'column_name 2']]`

# In[10]:


df_reduced = df[['customer_id', 'product_id', 'star_rating', 'product_title']]


# Check if you have duplicates after reducing the dataset. 

# In[11]:


duplicates = df_reduced[df_reduced.duplicated()]

len(duplicates)


# **Answer**: 131

# **Question:** Why do you have duplicates in your dataset now? What changed after you reduced the dataset? Review the first 20 lines of the duplicates.
# 
# **Hint**: Use the `pandas.head(<number>)` function to print the rows.

# In[12]:


duplicates.head(20)


# **Hint:** Take a look at the first two elements in the duplicates dataframe, and query the original dataframe df to see what the data looks like. You can use the `query` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)).
# 
# For example:
# 
# ```
# df_eg = pd.DataFrame({
#             'A': [1,2,3,4],
#             'B': [
#         })
# df_eg.query('A > 1 & B > 0')
# ```

# In[13]:


df.query("customer_id == 17570065 & product_id == 'B00R3EEO2G'")


# **Answer:** The dataset has duplicates because there are products with the same information but different `review_id` or `product_id`.

# Before continuing, remove the duplicate rows.
# 
# **Hint**: Use the `~` operator to select all the rows that aren't duplicated. For example:
#     
# ```
# df_eg = pd.DataFrame({
#             'A': [1,2,3,4],
#             'B': [2,0,5,2]
#         })
# df_eg[~(df_eg['B'] > 0)]
# ```

# In[14]:


df_reduced = df_reduced[~df_reduced.duplicated()]


# ### Visualize some of the rows in the dataset
# If you haven't done so in the above, you can use the space below to further visualize some of your data. Look specifically at the distribution of features like `star_rating`, `customer_id`, and `product_id`.
# 
# **Specific questions to consider**
# 
# 1. After looking at the distributions of features, to what extent might those features help your model? Is there anything you can deduce from those distributions that might be helpful in better understanding your data? 
# 
# 2. Should you use all the data? What features should you use?
# 
# 3. What month has the highest count of user ratings?
# 
# Use the cells below to visualize your data and answer these and other questions that might be of interest to you. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Include a summary of your answers to these and similar questions in your project presentations.</span>

# Use `sns.barplot` ([documentation](https://seaborn.pydata.org/generated/seaborn.barplot.html)) to plot the `star_rating` density and distribution.

# In[15]:


# Count the number of reviews with a specific rating
df['star_rating'].value_counts().reset_index()
sns.barplot(
    x='index', 
    y='star_rating', 
    data=_,  # The underscore symbol in Python is used to store the output of the last operation
    palette='GnBu_d'
)


# **Question:** What month contains the highest count of user ratings?
# 
# **Hint**:  
# 1. Use `pd.to_datetime` to convert the `review_date` column to a datetime column.  
# 2. Use the month from the `review_date` column. You can access it for a datetime column using `<column_name>.dt.month`.
# 3. Use the `groupby` function using `idxmax`.
# 

# In[16]:


# Convert the review date to a datetime type and count the number of ratings by month
df['review_date'] = pd.to_datetime(df.review_date)
df.groupby(df.review_date.dt.month).star_rating.count().reset_index()
sns.barplot(x='review_date', y='star_rating', data=_, palette='GnBu_d')


# In[17]:


max_month = df.groupby(df.review_date.dt.month).star_rating.count().idxmax()
print(f'The month with the most reviews is: {max_month}')


# **Bonus question (optional):** Which years have the most and least reviews?

# In[18]:


df.groupby(df.review_date.dt.year).star_rating.count().reset_index()
fig = plt.gcf()
fig.set_size_inches(10, 5)
sns.barplot(x='review_date', y='star_rating', data=_, palette='GnBu_d')


# **Answer:** The years with the least amount of data are 2000 and 2001, with only 1 review. The year with the highest number of reviews is 2015.

# ### Cleaning data

# **Question**: How heterogeneous are the number of reviews per customer and reviews per video? Use quantiles to find out.
# 
# **Hint**: Use `<dataframe>['columns_name'].value_counts()` for the customers and products dataframe, and use `<dataframe>.quantile(<list>)` to find the relationship.

# In[19]:


customers = df['customer_id'].value_counts()
products = df['product_id'].value_counts()

quantiles = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 
             0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 
             0.999, 1]
print('customers\n', customers.quantile(quantiles))
print('products\n', products.quantile(quantiles))


# **Answer:** Only about 5% of customers have rated 5 or more videos, and only 25% of videos have been rated by 9+ customers.

# Filter out this long tail. Select the customers that have rated 18 or more videos and the products that have more than 95 reviews.

# In[20]:


customers1 = customers[customers >= 18]
products1 = products[products >= 95]

reduced_df = (
    df_reduced.merge(pd.DataFrame({'customer_id': customers1.index}))
              .merge(pd.DataFrame({'product_id': products1.index}))
)


# **Question:** What is the shape of `customers1`, `products1`, and the new dataframe reduced_df?
# 
# **Note**: Use f-strings for this:
# 
# ```
# x= 3
# print(f'X = {x}')
# ```

# In[21]:


print(f'Number of users is {customers1.shape[0]} and number of items is {products1.shape[0]}.')
print(f'Length of reduced df is {reduced_df.shape[0]}.')


# Print the first 5 columns of the dataframe.

# In[22]:


reduced_df.head()


# **Question:** Does `reduced_df` maintain the same ratio of ratings?

# In[23]:


reduced_df['star_rating'].value_counts().reset_index()
sns.barplot(x='index', y='star_rating', data=_, palette='GnBu_d')


# **Answer:** The number of reviews with a 1-star rating decreased in proportion to 2-star ratings. That was not the case with the original data.

# Now, recreate the customer and product distributions of count per customer and product.  
# 
# **Hint**: Use the `value_counts()` function on the `customer_id` and `product_id` columns.

# In[24]:


customers = reduced_df['customer_id'].value_counts()
products = reduced_df['product_id'].value_counts()

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('Distribution of counts per customer and product')
sns.distplot(customers, kde=False, ax=axs[0], color='teal')
sns.distplot(products, kde=False, ax=axs[1])


# Next, number each user and item, giving them their own sequential index. This will allow you to hold the information in a sparse format where the sequential indices indicate the row and column in the ratings matrix.
# 
# To create the `customer_index` and `product_index`, create a new dataframe with `customer_id` as the index value and a sequential counter/values for the user and item number. Once you are finished creating both indexes, use the Pandas `merge` function to merge `customer_index` with `product_index`.
# 
# **Hint**: Use the `shape` function to generate the total number of customers and products. Use `np.arange` to generate a list of numbers from 0 to the number of customers and products.

# In[25]:


customer_index = pd.DataFrame({'customer_id': customers.index, 
                               'user': np.arange(customers.shape[0])})
product_index = pd.DataFrame({'product_id': products.index, 
                              'item': np.arange(products.shape[0])})

reduced_df = reduced_df.merge(customer_index).merge(product_index)
reduced_df.head()


# Sample answer:
# <div class="output_subarea"><div>
# 
# <table class="dataframe" border="1">
#   <thead>
#     <tr style="text-align: right">
#       <th></th>
#       <th>customer_id</th>
#       <th>product_id</th>
#       <th>star_rating</th>
#       <th>product_title</th>
#       <th>user</th>
#       <th>item</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>11763902</td>
#       <td>B00PSLQYWE</td>
#       <td>4</td>
#       <td>Downton Abbey Season 5</td>
#       <td>3065</td>
#       <td>103</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>1411480</td>
#       <td>B00PSLQYWE</td>
#       <td>5</td>
#       <td>Downton Abbey Season 5</td>
#       <td>130</td>
#       <td>103</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>35303629</td>
#       <td>B00PSLQYWE</td>
#       <td>5</td>
#       <td>Downton Abbey Season 5</td>
#       <td>4683</td>
#       <td>103</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>21285980</td>
#       <td>B00PSLQYWE</td>
#       <td>5</td>
#       <td>Downton Abbey Season 5</td>
#       <td>449</td>
#       <td>103</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>29260449</td>
#       <td>B00PSLQYWE</td>
#       <td>5</td>
#       <td>Downton Abbey Season 5</td>
#       <td>131</td>
#       <td>103</td>
#     </tr>
#   </tbody>
# </table>
# </div></div>

# # Project Solution Lab 3: 

# # Step 3: Model training and evaluation
# 
# There are some preliminary steps that you must include when converting the dataset from a dataframe to a format that a machine learning algorithm can use. For Amazon SageMaker, here are the steps you need to take:
# 
# 1. Split the data into `train_data` and `test_data`.    
# 2. Convert the dataset to an appropriate file format that the Amazon SageMaker training job can use. This can be either a CSV file or record protobuf. For more information, see [Common Data Formats for Training](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html). For this problem, the data will be sparse, so you can use the `scipy.sparse.lilmatrix` function and then convert the function to the `RecordIO protobuf` format using `sagemaker.amazon.common.write_spmatrix_to_sparse_tensor`.    
# 3. Upload the data to your Amazon S3 bucket. If you have not created one before, see [Create a Bucket](https://docs.aws.amazon.com/AmazonS3/latest/gsg/CreatingABucket.html).    
# 
# Use the following cells to complete these steps. Insert and delete cells where needed.
# 
# #### <span style="color: blue;">Project presentation: Take note of the key decisions you've made in this phase in your project presentations.</span>

# ### Prepare the data
# 
# You are at a point where you can start preparing the dataset as input for your model. Every model has different input needs. Some of the algorithms implemented in Amazon SageMaker require the data to be in the recordIO-wrapped protobuf form. You will take care of that in the following cells.
# 
# First, split the dataset into training and test sets. This will allow you to estimate the model's accuracy on videos that customers rated but that weren't included in the training.
# 
# Start with creation of the `test_df` dataframe. Create the dataframe by grouping the dataframe on `customer_id` and using the `last` function, similar to `pd.groupby('  ').last()`.

# In[26]:


test_df = reduced_df.groupby('customer_id').last().reset_index()


# To create the training data, remove the values present in `test_df` from the `reduced_df` dataframe.
# 
# **Hint**: Merge the `reduced_df` dataframe with the `test_df` dataset with `customer_id` and `product_id` columns as an outer join.

# In[27]:


train_df = reduced_df.merge(test_df[['customer_id', 'product_id']], 
                            on=['customer_id', 'product_id'], 
                            how='outer', 
                            indicator=True)
train_df = train_df[(train_df['_merge'] == 'left_only')].reset_index()


# In[28]:


test_df.head()


# Now you can look at some basic characteristics of the data that will later help you convert the features to an appropriate format for training your model.
# 
# Create two variables `nb_rating_test` and `nb_ratings_train` for the length of the test and training datasets.

# In[29]:


nb_ratings_test = len(test_df.index)
nb_ratings_train = len(train_df.index)
print(f" Training Count: {nb_ratings_train}")
print(f" Test Count: {nb_ratings_test}")


# ### Data conversion
# 
# Now, you can convert your Pandas dataframes into a sparse matrix. This process is the same for both train and test. The Amazon SageMaker implementation of factorization machines takes recordIO-wrapped protobuf, where the data you have today is a Pandas dataframe on disk. Therefore, you are going to convert the data to a sparse matrix to express the relationships between each user and each movie.

# In[30]:


# First convert data to sparse fomat with scipy lil_matrix

from scipy.sparse import lil_matrix

def loadDataset(df, lines, columns, regressor=True):
    """
    Convert the pandas dataframe into sparse matrix
    
    Args:
        df: DataFrame
        lines: number of rows of the final sparse matrix
        columns: number of columns of final sparse matrix
        regressor: Boolean value to check if we are using regression
                  or classification
    Returns:
        X: Feature vector
        Y: Label vector
    """
    # Features are one-hot encoded in a sparse matrix
    # Use scipy.sparse.lil_matrix to create the feature vector X of type float32
    X = lil_matrix((len(df), lines + columns)).astype('float32')
    
    # Labels are stored in a vector. Instantiate an empty label vector Y.
    Y = []
    
    line = 0
    
    # For each row in the dataframe, use 1 for the item and product number
    for index, row in df.iterrows():
        X[line,row['user']] = 1
        X[line, lines + (row['item'])] = 1
        line += 1

        if regressor:
            # If using regression, append the star_rating
            Y.append(row['star_rating'])
        else:
            # Use 1 for star_rating 5 else use 0
            if int(row['star_rating']) >= 5:
                Y.append(1)
            else:
                Y.append(0)
    
    # Convert the list into NumPy array of type float32  
    Y = np.array(Y).astype('float32')
    return X, Y


# Use the `loadDataset` function to create the training and test sets.

# In[31]:


print(customers.shape[0], 
      products.shape[0],
      customers.shape[0] + products.shape[0])

X_train, Y_train = loadDataset(train_df, customers.shape[0], 
                               products.shape[0])
X_test, Y_test = loadDataset(test_df, customers.shape[0], 
                             products.shape[0])


# Now that your data is in a sparse format, save it as a protobuf format and upload it to Amazon S3. This step might look intimidating, but most of the conversion effort is handled by the Amazon SageMaker Python SDK, imported as SageMaker below.

# In[32]:


import io 
import sagemaker.amazon.common as smac

def writeDatasetToProtobuf(X, bucket, prefix, key, d_type, Y=None):
    buf = io.BytesIO()
    if d_type == "sparse":
        smac.write_spmatrix_to_sparse_tensor(buf, X, labels=Y)
    else:
        smac.write_numpy_to_dense_tensor(buf, X, labels=Y)
        
    buf.seek(0)
    obj = '{}/{}'.format(prefix, key)
    boto3.resource('s3').Bucket(bucket).Object(obj).upload_fileobj(buf)
    return 's3://{}/{}'.format(bucket,obj)
    
fm_train_data_path = writeDatasetToProtobuf(X_train, bucket, prefix, 'train', "sparse", Y_train)    
fm_test_data_path  = writeDatasetToProtobuf(X_test, bucket, prefix, 'test', "sparse", Y_test)    
  
print("Training data S3 path: ", fm_train_data_path)
print("Test data S3 path: ", fm_test_data_path)


# You are finally finished with data preparation. Hooray! As you can see, it takes a lot of time and effort to clean and prepare the data for modeling. This is true for every single data science project, and this step has a high impact on the outcome. Make sure you spend enough time understanding and preparing your data for training in all future machine learning dventures!

# ## Training the model
# 
# Now it's time to train the model. You will use an Amazon SageMaker training job for that. Amazon SageMaker training jobs are an easy way to create models, as you don't really have to write all the code for training. That is already handled for you in a nice container format.
# 
# The general workflow for creating training jobs from the notebook is to instantiate the predictor, pass some hyperparameters, and then pass the data in the correct format. This is what happens in the following cell.
# 
# For more more information about FM estimator, see [FactorizationMachines](https://sagemaker.readthedocs.io/en/stable/factorization_machines.html).
# 
# For more information about hyperparameters, see [Factorization Machines Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines-hyperparameters.html).
# 
# **Hint**: Example:
# 
# ```
# sess = sagemaker.Session()
# 
# pca = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
#                                     role,
#                                     instance_count=1,
#                                     instance_type='ml.m4.xlarge',
#                                     output_path=output_location,
#                                     sagemaker_session=sess)
#                                     
# pca.set_hyperparameters(featuer_dim=50000,
#                         num_components=10,
#                         subtract_mean=True,
#                         algorithm_mode='randomized',
#                         mini_batch_size=200)
#                         
# pca.fit({'train': s3_train_data})
# ```

# In[33]:


from sagemaker import get_execution_role
from sagemaker import image_uris

output_prefix = 's3://' + bucket + '/sagemaker-fm/model'
instance_type='ml.m4.xlarge'
batch_size = 128

fm = sagemaker.estimator.Estimator(
    image_uris.retrieve("factorization-machines",boto3.Session().region_name),
    role, 
    instance_count=1, 
    instance_type=instance_type,
    output_path=output_prefix,
    sagemaker_session=sagemaker.Session()
)

fm.set_hyperparameters(feature_dim=X_train.shape[1],
                       predictor_type='regressor',
                     # predictor_type='binary_classifier',
                       mini_batch_size=batch_size,
                       num_factors=64,
                       epochs=25,
                       clip_gradient=5.0,
                       rescale_grad=1.0/batch_size)

fm.fit({'train': fm_train_data_path, 'test': fm_test_data_path})


# **Question:** What does changing the `batch_size` and `epochs` do to the final metric?  
# 
# **Answer:** There are a couple of things to notice here. The `predictor_type` is set to `'regressor'` because you are trying to predict the star rating. The `batch_size` is set to 128. This value can be tuned for relatively minor improvements in fit and speed, but selecting a reasonable value relative to the dataset is appropriate in most cases. `num_factors` is set to 64. As mentioned initially, factorization machines find a lower dimensional representation of the interactions for all features. Making this value smaller provides a more parsimonious (simple) model, closer to a linear model, but may sacrifice information about interactions. Making it larger provides a higher dimensional representation of feature interactions but adds computational complexity and can lead to overfitting. In a practical application, time should be invested to tune this parameter to the appropriate value.

# **Question:** Check the output of the model. What is the meaning of the metrics used? Is there a difference between the training and testing sets? If yes, what is the meaning of that?  

# **Answer:** The **MSE**, **RMSE**, and **Absolute loss** are considerably lower in the training set than in the testing dataset. This might indicate that the model is being molded to the pattern of the training set and not the general pattern on the reviews. This phenomenon is called overfitting.

# ### Evaluate
# 
# Congratulations! You have successfully launched an Amazon SageMaker training job. Now what? Well, you need a way to verify that your model is actually predicting coherent values. How do you do this?
# 
# Start by calculating a naive baseline to approximate how well your model is doing. The simplest estimate would be to assume every user item rating is just the average rating over all ratings. This is basically saying that you have a model that only learned to output the mean value of all reviews.
# 
# **Note:** You could do better by using each individual video's average; however, in this case, it doesn't really matter because the same conclusions would hold.

# Calculate the mean of `star_rating` to get the `naive_guess`. Then, calculate the naive MSE by squaring the naive guess from the test `star_rating` and getting an average.
# 
# $average(test(star\_rating) - naive\_guess)^2)$

# In[34]:


naive_guess = np.mean(train_df['star_rating'])
print('Naive MSE:', np.mean((test_df['star_rating'] - naive_guess) ** 2))


# Now, calculate predictions for your test dataset. To this end, you'll need to _deploy_ the model you just trained.
# 
# **Note:** This will align closely to your CloudWatch output above but may differ slightly due to skipping partial mini-batches in the `eval_net` function.
# 
# Use `<estimator_name>.deploy` with `initial_instance_count=1, instance_type=ml.m4.xlarge`.

# In[35]:


fm_predictor = fm.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


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
# #### <span style="color: blue;">Project presentation: Record questions to these and other similar questions you might answer in this section in your project presentations. Record key details and decisions you've made in your project presentations.</span>

# The deployment process involves creating an instance of the specified size, in this case `ml.m4.xlarge`, with the model you trained and saved on Amazon S3. To get a prediction, you need to pass your data in a serialized form of JSON. The output you get from the inference will be in serialized JSON form as well, so you also need to deserialize it to get the predicted values.

# import json
# from sagemaker.deserializers import JSONDeserializer
# from sagemaker.serializers import BaseSerializer
# class fm_serializer(BaseSerializer):
#     CONTENT_TYPE='application/json'
#     def serialize(data):
#             js = {'instances': []}
#             for row in data:
#                 js['instances'].append({'features': row.tolist()})
#             return json.dumps(js)
# fm_predictor.serializer = fm_serializer
# fm_predictor.deserializer = JSONDeserializer()
# print(f"Accepted content type: {fm_predictor.content_type}")

# In[36]:


# Create a serializer function for the predictor
import json
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import BaseSerializer

class fm_serializer(BaseSerializer):
    CONTENT_TYPE='application/json'
    def serialize(data):
            js = {'instances': []}
            for row in data:
                js['instances'].append({'features': row.tolist()})
            return json.dumps(js)
fm_predictor.serializer = fm_serializer
fm_predictor.deserializer = JSONDeserializer()
print(f"Accepted content type: {fm_predictor.content_type}")


# Check how your training set did. Use the endpoint to get predictions from your model.
# 
# First, look at what a single prediction looks like.

# Amazon SageMaker model containers must respond to requests within 60 seconds. The model itself can have a maximum processing time of 60 seconds before responding to the /invocations. To do that, call the `predict` function for 5 rows at a time and then add those rows to a list. 

# In[37]:


# Pass the X_train data to the predictor deployed 
ytrain_p = []
for i in range(0, 1000, 5):
    preds = fm_predictor.predict(X_train[i:i + 5].toarray())['predictions']
    p = [ytrain_p.append(x['score']) for x in preds]


# **Question:** Now that you have inferences, do a sanity check. What are the minimum and maximum values predicted in the inferences? Do those correspond to the minimum and maximum values in the training data?

# In[38]:


print('The minimum rating predicted is: ', min(ytrain_p), 'and the maximum is: ',max(ytrain_p))


# Now, check your test dataset.

# In[39]:


Y_pred = []
for i in range(0, X_test.shape[0], 5):
    preds = fm_predictor.predict(X_test[i:i+5].toarray())['predictions']
    p = [Y_pred.append(x['score']) for x in preds]


# **Question:** How are the min and max values alike in the predictions? Bonus point if you check the entire distribution (histogram).

# In[40]:


max(Y_pred), min(Y_pred)


# In[41]:


sns.distplot(Y_pred, kde=False, bins=4)


# Finally, calculate the mean squared error for the test set and see how much of an improvement it is from the baseline.

# In[42]:


print('MSE:', np.mean((test_df['star_rating'] - Y_pred) ** 2))


# For recommender systems, subjective accuracy also matters. Get some recommendations for a random user to see if they make intuitive sense.
# 
# Try using user number 200, and see what they have watched and rated highly.

# In[43]:


reduced_df[reduced_df['user'] == 200].sort_values(
    ['star_rating', 'item'], ascending=[False, True])


# As you can see, this user likes to watch comedies, romance, and light-hearted movies and dislikes drama and fantasy movies. Let's see how your model predicts movie ratings for this user.

# In[44]:


def prepare_predictions(user_id, number_movies, columns):
    # Create the sparse matrix 
    X = lil_matrix((number_movies, columns)).astype('float32')
    movie_index_start = columns - number_movies

    # Fill out the matrix. Each row will be the same user with every possible movie.
    for row in range(number_movies):
        X[row, user_id - 1] = 1
        X[row, movie_index_start + row] = 1

    return X

user_200 = prepare_predictions(200, products.shape[0], customers.shape[0] + products.shape[0])


# Now create a list of all the ratings that the model would predict for user 200 for all movies.

# In[45]:


pred_200 = []
for i in range(0, user_200.shape[0], 5):
    preds = fm_predictor.predict(user_200[i:i+5].toarray())['predictions']
    p = [pred_200.append(x['score']) for x in preds]


# Now loop through and predict user 200's ratings for every common video in the catalog to see which ones to recommend or not recommend. 
# 
# Create a new dataframe `titles` by using the `reduced_df` dataframe to group by the items. Use the `product_title` column and create another column `score` and add the values from `pred_200` to it.

# In[46]:


titles = reduced_df.groupby('item')['product_title'].first().reset_index()
titles['score'] = pred_200


# **Question:** What products got the highest score?  
# 
# **Hint**: Use the `sort_values` function to sort columns `score` and `item` and use parameter `asecnding=[False,True]`

# In[47]:


titles.sort_values(['score', 'item'], ascending=[False, True])


# **Question:** What can you conclude from the highly rated and lowest rated shows for the user?

# **Answer:** The predicted highly rated shows have some well-reviewed TV dramas and some book adaptations. The lowest scores are from comedies, child, and teenage movies.  

# See if your recommendations have correlations with other users. Try user 201. Perform the same operations as you did for user 200.

# In[48]:


user_201 = prepare_predictions(201, products.shape[0], customers.shape[0] + products.shape[0])

pred_201 = []
for i in range(0, user_201.shape[0], 5):
    preds = fm_predictor.predict(user_201[i:i+5].toarray())['predictions']
    p = [pred_201.append(x['score']) for x in preds]


# In[49]:


plt.scatter(pred_200, pred_201)
plt.show()


# **Question:** What can you conclude from the scatter plot between the two users?  

# **Answer:** This correlation is nearly perfect. Essentially, the average rating of items dominates across users, and our system will recommend the same well-reviewed items to everyone. That basically means that the model is only finding the best rated shows and presenting them to all users. Now we need to see if we can do better by shifting the paradigm from regression to classification.

# Delete the endpoint you created for inference because you won't be using it anymore.

# In[50]:


sagemaker.Session().delete_endpoint(fm_predictor.endpoint_name)


# # Project Solution Lab 4: 

# # Iteration II
# 
# # Step 4: Feature engineering
# 
# You've now gone through one iteration of training and evaluating your model. Given that the outcome you reached for your model the first time probably wasn't sufficient for solving your business problem, what are some things you could change about your data to possibly improve model performance?
# 
# ### Key questions to consider:
# 1. How might changing the machine learning problem help your dataset? You tried to use regression to solve the problem; can classification help?
# 2. What do you need to do to change the machine learning problem to a machine learning classification problem? Write down the new problem statement for classification.
# 
# #### <span style="color: blue;">Project presentation: Record key decisions and methods you use in this section in your project presentations, as well as any new performance metrics you obtain after evaluating your model again.</span>

# Now change the training datasets to have a binary output depending on the rating they get. You will consider recommending something to a user when the rating is 5 stars, and you will save again as a protobuf format in Amazon S3. Do the following:  
# 
# 1. Use the `loadDataset` function with the option `regression=False` to create your training datasets.   
# 2. Write the dataset as a protobuf format.  
# 3. Retrain the model using `predictor_type='binary_classifier'`.   
# 4. Deploy your model to an endpoint and evaluate the model, similar to how you did before on the test set.   
# 5. Inspect how you did on the test set using a confusion matrix.   

# In[51]:


X_train_class, Y_train_class = loadDataset(train_df, customers.shape[0], 
                               products.shape[0], regressor=False)
X_test_class, Y_test_class = loadDataset(test_df, customers.shape[0], 
                             products.shape[0], regressor=False)


# In[52]:


# Write dataset as a protobuf

fm_train_data_path = writeDatasetToProtobuf(X_train_class, bucket, prefix, 'train_class', "sparse", Y_train_class)    
fm_test_data_path  = writeDatasetToProtobuf(X_test_class, bucket, prefix, 'test_class', "sparse", Y_test_class)    
  
print("Training data S3 path: ", fm_train_data_path)
print("Test data S3 path: ", fm_test_data_path)


# Finally, retrain the model, changing from regression to binary classification. Use the same code and settings that you did when you trained your model previously, but change the `predictor_type='binary_classifier`.

# In[53]:


from sagemaker import get_execution_role
from sagemaker import image_uris

#output_prefix= 's3://<LabBucketName>/sagemaker-fm/model'

output_prefix = 's3://' + bucket + '/sagemaker-fm/model'
instance_type='ml.m4.xlarge'
batch_size = 512

fm = sagemaker.estimator.Estimator(
    image_uris.retrieve("factorization-machines",boto3.Session().region_name),
    role, 
    instance_count=1, 
    instance_type=instance_type,
    output_path=output_prefix,
    sagemaker_session=sagemaker.Session()
)

fm.set_hyperparameters(feature_dim=X_train.shape[1],
                     # predictor_type='regressor',
                       predictor_type='binary_classifier',
                       mini_batch_size=batch_size,
                       num_factors=128,
                       epochs=25,
                       clip_gradient=5.0,
                       rescale_grad=1.0/batch_size
                       )

fm.fit({'train': fm_train_data_path, 'test': fm_test_data_path})


# Evaluate the performance of this new model. Deploy the model, determine a serializer, and then pass the test data.

# In[54]:


from sagemaker import deserializers
fm_predictor = fm.deploy(initial_instance_count=1, 
                         instance_type='ml.m4.xlarge', 
                         serializer=fm_serializer, 
                         deserializer=JSONDeserializer())


# fm_predictor.content_type = 'application/json'
# fm_predictor.serializer = fm_serializer
# fm_predictor.deserializer = deserializers.JSONDeserializer()

# In[55]:


# Pass the testing data to the classifier and get all the predictions
Y_pred = []
for i in range(0, X_test_class.shape[0], 5):
    preds = fm_predictor.predict(X_test_class[i:i+5].toarray())['predictions']
    p = [Y_pred.append(x['score']) for x in preds]


# #### Inspect the results
# 
# To inspect how well the classifier is doing, calculate and plot a confusion matrix. Use the implementation from **Scikit-Learn**.

# In[56]:


from sklearn.metrics import confusion_matrix


# In[57]:


true = Y_test_class.astype(int)
predicted = [1 if value > 0.5 else 0 for value in Y_pred]
conf_matrix = confusion_matrix(true, predicted)
print(conf_matrix)
sns.heatmap(conf_matrix)


# In[58]:


# tn, fp, fn, tp
conf_matrix.ravel()


# **Question:** What is the accuracy of your model?  
# 
# **Hint**:
# $$ Accuracy = \frac{TP + TN}{TP + FP + FN + TN} $$

# In[59]:


# Accuracy
(conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()


# **Question:** How did your model do compared to a naive baseline model of predicting everything as 1?

# In[60]:


(reduced_df.star_rating > 4).value_counts() / reduced_df.shape[0] * 100


# **Answer:** This model is doing significantly better than the naive model, which randomly generates an output (~50% accuracy). If you look closely to the confusion matrix, the model is being conservative and preferring not to recommend things to users. That  means most of the mistakes the model is making come from not recommending movies to users.

# In[61]:


# Delete inference endpoint
sagemaker.Session().delete_endpoint(fm_predictor.endpoint_name)


# ## Combining powers with KNN
# 
# You saw that your classifier model is doing a better job than the regressor model. Now, see if you can repackage it to fit a k-nearest neighbor (KNN) model to predict the *k closest* items to the one a customer likes and then recommend those, instead of predicting the ratings (regressor) or whether a user would like a movie or not (binary classification).
# 
# Start by downloading the model from Amazon S3. Then, repackage it to fit a KNN model.
# 
# **Note:** Make sure the kernel you are using is `conda_mxnet_p36` so you can run the next cells.

# ### Download model data

# In[62]:


import mxnet as mx
model_file_name = 'model.tar.gz'
model_full_path = f'{fm.output_path}/{fm.latest_training_job.job_name}/output/{model_file_name}'
print(f'Model Path: {model_full_path}')

# Download FM model 
os.system('aws s3 cp ' + model_full_path + ' .')

# Extract model file for loading to MXNet
os.system('tar xzvf ' + model_file_name)
os.system('unzip -o model_algo-1')
os.system('mv symbol.json model-symbol.json')
os.system('mv params model-0000.params')


# ### Extract model data to create item and user latent matrixes

# Now you are going to extract the values that represent each user and item after training a factorization machine. The result of the training is two matrices that, when multiplied together, will represent the target values (zero or one) as closely as possible.
# 
# In more mathematical terms, factorization machines model output consists of three N-dimensional arrays (ndarrays):
# 
#     V – a (N x k) matrix, where:
#         k is the dimension of the latent space
#         N is the total count of users and items
#     w – an N-dimensional vector
#     b – a single number: the bias term
# 
# To extract these values, which you will use as features, you need to first load the model. Then, extract the values of each of the three matrices and build the `knn_item_matrix` and t`knn_user_matrix` matrices.

# In[63]:


# Extract model data
m = mx.module.Module.load('./model', 0, False, label_names=['out_label'])
V = m._arg_params['v'].asnumpy()
w = m._arg_params['w1_weight'].asnumpy()
b = m._arg_params['w0_weight'].asnumpy()

nb_users = customers.shape[0]
nb_item = products.shape[0]

# Item latent matrix - concat(V[i], w[i]).  
knn_item_matrix = np.concatenate((V[nb_users:], w[nb_users:]), axis=1)
knn_train_label = np.arange(1,nb_item+1)

# User latent matrix - concat (V[u], 1) 
ones = np.ones(nb_users).reshape((nb_users, 1))
knn_user_matrix = np.concatenate((V[:nb_users], ones), axis=1)


# ## Building KNN model
# 
# Now that you have the training data, you can now feed it to a KNN model. As you did before, you need to save the protobuf IO formatted data to Amazon S3, instantiate the model, and set the hyperparameters.
# 
# Start by setting up the path and the estimator.

# In[64]:


print('KNN train features shape = ', knn_item_matrix.shape)
knn_prefix = 'knn'
train_key = 'train_knn'
knn_output_prefix  = f's3://{bucket}/{knn_prefix}/output'
knn_train_data_path = writeDatasetToProtobuf(knn_item_matrix, bucket, 
                                             knn_prefix, train_key, 
                                             "dense", 
                                             knn_train_label)
print(f'Uploaded KNN train data: {knn_train_data_path}')

nb_recommendations = 100

# Set up the estimator
knn = sagemaker.estimator.Estimator(
    image_uris.retrieve("knn",boto3.Session().region_name),
    get_execution_role(),
    instance_count=1,
    instance_type=instance_type,
    output_path=knn_output_prefix,
    sagemaker_session=sagemaker.Session()
)


# Now, you will set the hyperparameters. Note that this approach uses the default `index_type` parameter for KNN. It is precise but can be slow for large datasets. In such cases, you may want to use a different `index_type` parameter leading to an approximate, yet faster answer.
# 
# For more information about index types, see [k-NN Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/kNN_hyperparameters.html).

# In[65]:


knn.set_hyperparameters(feature_dim=knn_item_matrix.shape[1], 
                        k=nb_recommendations, 
                        index_metric="INNER_PRODUCT", 
                        predictor_type='classifier', 
                        sample_size=200000)


knn.fit({'train': knn_train_data_path})


# Now that you have a trained model, save it so you can reference it for batch inference.

# In[66]:


knn_model_name =  knn.latest_training_job.job_name
print("created model: ", knn_model_name)

# Save the model so that you can reference it in the next step during batch inference
sm = boto3.client(service_name='sagemaker')
primary_container = {
    'Image': knn.image_uri,
    'ModelDataUrl': knn.model_data,
}

knn_model = sm.create_model(
        ModelName = knn.latest_training_job.job_name,
        ExecutionRoleArn = knn.role,
        PrimaryContainer = primary_container)
print("saved the model")


# ## Batch transform
# 
# To see the predictions your model made, you would have to create inferences and see if they make sense. You could repeat the process as last time and check one user at a time with all possible combinations of items. However, Amazon SageMaker provides a batch transform job that you can use to do inference over the entire dataset. For more information, see [Get Inferences for an Entire Dataset with Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html).
# 
# In this section, you will use a batch transform to predict the top 100 recommendations for all the users.

# In[67]:


# Upload inference data to S3
knn_batch_data_path = writeDatasetToProtobuf(knn_user_matrix,
                                             bucket, 
                                             knn_prefix, 
                                             train_key, 
                                             "dense")
print ("Batch inference data path: ",knn_batch_data_path)

# Initialize the transformer object
transformer =sagemaker.transformer.Transformer(
    base_transform_job_name="knn",
    model_name=knn_model_name,
    instance_count=1,
    instance_type=instance_type,
    output_path=knn_output_prefix,
    accept="application/jsonlines; verbose=true",
    
)

# Start a transform job
transformer.transform(knn_batch_data_path, 
                      content_type='application/x-recordio-protobuf',
                      split_type='RecordIO')
transformer.wait()


# You are now free to examine the predictions. Download them first.

# In[68]:


# Download predictions 
results_file_name = "inference_output"
inference_output_file = "knn/output/train_knn.out"
s3_client = boto3.client('s3')
s3_client.download_file(bucket, inference_output_file, results_file_name)


# In[69]:


# Open file and load it to memory
with open(results_file_name) as f:
    results = f.readlines() 


# The results contain the 100 nearest neighbor movie IDs with their corresponding distances. See how it looks for user number 200.

# In[70]:


test_user_idx = 200
u_one_json = json.loads(results[test_user_idx])
recommended_movies = [int(movie_id) for movie_id in u_one_json['labels']]
distances = [round(distance, 4) for distance in u_one_json['distances']]

print(f'Recommended movie Ids for user #{test_user_idx} : {recommended_movies}')

print(f'Movie distances for user #{test_user_idx} : {distances}')


# You got the movies closest to user 200's tastes. Now, you can see the titles.

# In[71]:


titles_200 = reduced_df[reduced_df.item.isin(recommended_movies)].product_title.unique()
titles_200


# Compare them with the favorite movies for user 200.

# In[72]:


reduced_df.query('user==200 & star_rating == 5')


# **Question:** Do you think these recommendations make sense? Explain why or why not.

# **Answer:** From a subjective point of view, the recommendations make sense. If you compare them with the top 100 suggestions from the first regression model, you'll see they only have three elements in common.

# In[73]:


np.isin(titles_200, titles.tail(100).product_title.unique()).sum()


# **Super bonus question:** Recover the predictions for user 201, and see how they compare with user 200. Are they still correlated? Do you think this approach was an improvement over the first regressor?

# In[74]:


# Recover the predictions for user 201

test_user_idx = 201
u_one_json = json.loads(results[test_user_idx])
recommended_movies_201 = [int(movie_id) for movie_id in u_one_json['labels']]


# In[75]:


# Print out recommendations

titles_201 = reduced_df[reduced_df.item.isin(recommended_movies_201)].product_title.unique()
titles_201


# In[76]:


# Compare the two predictions

overlap = np.isin(titles_200, titles_201).sum()
print(f'The recommendations for "user 201" that are present in "user 200" are: {overlap} out of: {len(titles_200)}')


# In[77]:


# Compare with user 201 likes

reduced_df.query('user==201 & star_rating == 5')


# In[78]:


test_user_idx = 900
u_one_json = json.loads(results[test_user_idx])
recommended_movies_900 = [int(movie_id) for movie_id in u_one_json['labels']]
titles_900 = reduced_df[reduced_df.item.isin(recommended_movies_201)].product_title.unique()
overlap_900 = np.isin(titles_200, titles_900).sum()
print(f'The recommendations for "user 900" that are present in "user 200" are: {overlap} out of: {len(titles_200)}')
reduced_df.query('user==900 & star_rating == 5')


# **Answer:** This final model is again recommending the same films to different users. That means there is a big cluster of movies that the KNN dims as close neighbors, and this cluster is dominating the rest. The thing about recommendation is that it is a subjective matter. You can get a sense of the taste of one person. but at the same time that person doesn't only like one genre. Most of the users have at least one comedy or animation movie in combination with their usual type of preference.

# There are a number of things you can do to improve these models, such as adding features besides rating, trying different feature selection, hyperparameter tuning, and changing the models. The most sophisticated recommendation algorithms are based on deep learning. This can also be explored.

# That is it! You now have a working recommender system that can tell you the top 100 movies for a user. Feel free to optimize and play with the hyperparameters and data to see if you can create an even better recommender system.

# ## Final thoughts
# 
# In this notebook, you used different techniques to create a recommendations system using only Amazon SageMaker built-in algorithms. You learned how to prepare data in different formats and do feature engineering. You were able to identify problems with your trained models and reframe the problem in different ways to achieve an end result. 
# 
# As you can see now, training a model requires a lot of steps, preparation, and validation. It is not a streamlined process but an iterative one. You can think of this as a virtuous cycle that usually has the following steps:
# 
# - Define the (business) problem.
# - Frame the problem as a machine learning problem.
# - Prepare data and perform freature engineerin.
# - Train and evaluate the model.
# - Deploy the model (inference).
# - Monitor and evaluate.
# 
# Every step has its own challenges, and each of the steps feeds each other. So it is important to pay attention to the entire pipeline, not only the model training.
# 
