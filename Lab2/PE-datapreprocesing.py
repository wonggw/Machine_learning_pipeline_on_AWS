#!/usr/bin/env python
# coding: utf-8

# # Practice Exercise – Preprocessing
# 
# The first rule in preprocessing is: **know your data!**
# 
# Therefore, you are going to explore your data over several dimensions and views to bring your raw data to a state of processed data, ready to be used for your model.
# 
# The **TBLF** approach is suggested:
# + **Try:** Explore the problem (in this case, explore your data)
# + **Broken?:** What is broken? (Do you see something wrong in your data)
# + **Learn:** Why is it wrong? (What is wrong with this data? How can you solve it?)
# + **Fix:** Fix the problem (What can I do to bring the wrong data to an acceptable solution?)

# In[ ]:





# ## Business scenario
# You are in charge of preprocessing the data your company wants to use in a new machine learning model. The model will be used to predict average influence of a journal, based on journal/publishing information coming from a variety of sources, including researchers, web scraping, and the publishers themselves.
# 
# You've received raw data related to the business problem to solve. Your task is to better understand the data using some descriptive statistics. Once you understand the data, you will clean and reshape it. You will use the final processed dataset in your model. 

# ## Learning objectives
# 1. Understand a business scenario and corresponding dataset by analyzing the data using descriptive statistics
# 2. Use visualization tools to support this analysis: 
#     - Scatter plots to spot correlations between features 
#     - Box and whisker plots and histograms to understand the distribution of your data
# 3. Use statistics tools to support previous analysis, such as a correlation matrix to quantify those relationships
# 4. Based on the analysis conclusions, prepare a processed dataset for the model by:
#     - Dealing with outliers 
#     - Dealing with missing values
#     - Cleaning the data

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', None)


# ## Dataset 
# The data comes from a variety of sources, including researchers, web scraping, and the publishers themselves. The data has been manipulated to be suited for this preprocessing task.

# ## Data schema
# A dataset on journal/publisher information with estimated-article-influence-scores:
# + journal_name: Name of the journal
# + issn: Unique publication code
# + citation_count_sum: Sum of the number of citations for each journal 
# + paper_count_sum: Sum of the number of papers for each journal
# + avg_cites_per_paper: Average number of citations per paper
# + ranking: Artificial label created for the exercise (possible values: A, B, C, D)
# + proj_ai: Projected average influence

# Start by loading the file.

# In[2]:


df_journals = pd.read_csv("estimated-article-influence-scores-Exerc2.csv", sep=";")
del df_journals["Unnamed: 0"]


# ## First glance at your data
# 
# #### Learn
# How many features do you have?

# In[3]:


print("Number of features: {}".format(df_journals.shape[1]))


# How many samples do you have?

# In[4]:


print("Number of samples: {}".format(df_journals.shape[0]))


# > ### Question 1
# > #### Try
# > Print the first 10 rows to see what the data looks like. Apply the `head` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html?highlight=head#pandas.DataFrame.head)) to your dataset.

# In[5]:


# Enter your code here
df_journals.head(10)


# ## Dealing with missing values
# #### Learn
# Check how many values are missing for each column.

# The **isnull()** command returns **True** for each cell of the dataset that is missing a value.

# In[6]:


df_journals.isnull().head(10)


# #### Is it broken?
# This visualization is not practical for big datasets. You can try to sum all rows for each column. As all the values above are booleans, if you sum by column (feature), you will have the amount of **true** values--the number of missing values for each feature.

# > ### Question 2
# > #### Fix
# > Apply the **.isnull()** function to your dataset as above, but concatenate the result with a **.sum()** function to show the number of missing values for each feature.<br/>
# 
# >`isnull` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isnull.html?highlight=isnull#pandas.DataFrame.isnull))<br/>
# >`sum` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html?highlight=sum#pandas.DataFrame.sum))

# In[7]:


# Enter your code here
df_journals.isnull().sum()


# #### Learn
# You observed that there are missing values in the dataset.

# Missing values are an important issue. Most models won't deal well with missing values.
# 
# You can remove the missing values or impute values for them. Each choice has pros and cons, depending on the importance of the particular feature for your training job, if you can afford to remove those missing values, etc.
# 
# The following command filters only the rows with `any` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.any.html?highlight=any#pandas.DataFrame.any)) missing value. The first 10 rows display.

# In[8]:


df_journals_null_data = df_journals[df_journals.isnull().any(axis=1)]
df_journals_null_data.head(10)


# ## Dealing with missing values
# #### Fix
# Let's suppose you have decided to remove the rows with missing values.
# 
# Remove the missing values by applying the `dropna` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html?highlight=dropna#pandas.DataFrame.dropna)) to your dataset. Save the result in a new dataset named **df_journals_no_miss**. Use the `shape` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html?highlight=shape#pandas.DataFrame.shape)) after that to confirm that your dataset has fewer rows.

# #### Learn
# How many rows are left?

# In[9]:


df_journals_no_miss = df_journals.dropna()
df_journals_no_miss.shape


# #### Learn
# Confirm that you have removed all missing values.

# In[10]:


df_journals_no_miss.isna().any()


# Now, suppose that you cannot afford to remove the rows with missing values. Maybe you don't want to reduce your already small dataset. 

# ## Imputing numerical values
# #### Learn
# One way to impute numerical values is to consider the mean of the feature (column) value. The following command calculates the mean of all numerical features in your dataset.

# In[11]:


citation_count_sum_MEAN = df_journals["citation_count_sum"].mean()
paper_count_sum_MEAN = df_journals["paper_count_sum"].mean()
avg_cites_per_paper_MEAN = df_journals["avg_cites_per_paper"].mean()
proj_ai_MEAN = df_journals["proj_ai"].mean()
print("citation_count_sum_MEAN: {}".format(citation_count_sum_MEAN))
print("paper_count_sum_MEAN: {}".format(paper_count_sum_MEAN))
print("avg_cites_per_paper_MEAN: {}".format(avg_cites_per_paper_MEAN))
print("proj_ai_MEAN: {}".format(proj_ai_MEAN))


# The following command updates each missing value with the calculated mean for the feature.
# 
# **Note:** The `inplace` clause ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html?highlight=fillna#pandas.DataFrame.fillna)) means that the update acts on the original dataset, so you don't need to attribute the result on a new dataset.

# In[12]:


df_journals["citation_count_sum"].fillna(citation_count_sum_MEAN, inplace=True)
df_journals["paper_count_sum"].fillna(paper_count_sum_MEAN, inplace=True)
df_journals["avg_cites_per_paper"].fillna(avg_cites_per_paper_MEAN, inplace=True)
df_journals["proj_ai"].fillna(proj_ai_MEAN, inplace=True)


# #### Learn
# Confirm that you have imputed for all of the numerical values. You will deal with categorical values next.

# In[13]:


df_journals.isna().any()


# ## Imputing categorical values
# #### Learn
# For categorical value imputation, a common approach is to use the most frequent value (the mode).

# #### Try
# Check the possible values for the "ranking" feature.
# 
# Pandas has an interesting command that prints the frequency for categoricals: `value_counts` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html?highlight=value_counts#pandas.Series.value_counts)).
# 
# Apply this command to the feature in your dataset to see the categoricals distribution of "ranking".

# In[14]:


df_journals["ranking"].value_counts()


# Now, check for the most frequent value for the categorical feature. Use the `mode` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mode.html?highlight=mode#pandas.DataFrame.mode)).
# 
# **Note:** You can access the feature using `df_journals["ranking"]` or `df_journals.ranking`.

# In[15]:


df_journals.ranking.mode()[0]


# #### Learn
# The command returned **D** as the most frequent value (the mode).

# Use the same command that you used to impute the numerical features to impute the most frequent ranking **D**.

# In[16]:


df_journals["ranking"].fillna('D', inplace=True)


# #### Learn
# Now, run the following command to see if you have any null values left. You should only have **False** values now.

# In[17]:


df_journals.isna().any()


# #### Learn
# To make sure the missing values turned into D, the **value_counts()** command should reflect that the new count for the categorical value **D** is the sum of the original one and the number of null values.

# In[18]:


df_journals["ranking"].value_counts()


# #### Is it broken?
# Unbalanced labels occur when the distribution between the labels presents some of them with much higher frequency than the others. For this dataset, it is not the case.

# ## Basic statistics for numerical values
# #### Exploring distributions
# ##### Try
# Calculate the min value, max value, mean, standard deviation, and the 25% and 75% percentiles for each column. Use the `describe` command ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html?highlight=describe#pandas.DataFrame.describe)).

# In[19]:


df_journals.describe()


# >### Question 3
# Plot the distribution of the "avg_cites_per_paper" feature. Use the `.plot.hist(bins=100)` function from matplotlib to plot the distribution of each feature.
# 
# >`hist` ([documentation](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html?highlight=hist#matplotlib.pyplot.hist))

# In[20]:


# Enter your code here
df_journals["avg_cites_per_paper"].plot.hist(bins=100)


# ## Box and whisker plots
# Box and whisker plots are great for spoting outliers. Let's explore using a box and whisker plot.

# >### Question 4
# >#### Try
# >The same way you've plotted the distribution, now make a box and whisker plot using the `boxplot` function ([documentation](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html?highlight=boxplot#matplotlib.pyplot.boxplot)) for the numerical feature "avg_cites_per_paper". 

# In[21]:


# Enter your code here
df_journals.boxplot(["avg_cites_per_paper"])


# #### Learn 
# Notice that the box and whisker plot helps to spot outliers.

# #### Learn
# Now, let's use a more interesting technique to remove outliers using the percentiles.
# 
# To print the percentiles of the dataset's numerical values, use the **.describe()** command.

# In[22]:


df_journals.describe()


# You will use the 25%, 50%, and 75% percentiles (or quartiles) as a guideline to remove the outliers.
# 
# For each feature, you will remove all data points outside the interquartile range (IQR), where:
# 
# $$IQR = 75\% \space quartile - 25\% \space quartile$$
# $$Lower \space Threshold = 50\% \space percentile - 1.5 * IQR$$
# $$Upper \space Threshold = 50\% \space percentile + 1.5 * IQR$$
# 
# The NumPy package has the `percentile` function ([documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.percentile.html?highlight=percentile#numpy.percentile)) that can help you to get the threshold values.

# >### Question 5 (Optional)
# >##### Try
# >**This question is optional and will not be graded**, but it is a nice way to exercise your curious side!
# 
# >Implement the following algorithm using the NumPy **percentile** function. 

# #### Fix
# Do the following for the "citation_count_sum" feature:
# + Use the guidelines and **percentile** function above to calculate the thresholds for each feature.
# + Remove the outliers outside this range for each feature.
# + Plot the box and whisker plot again to see the results.
# + Use the **shape** command before and after removal to see how many rows were removed.

# In[23]:


# Enter your code here
print("Number of samples before: {}".format(df_journals.shape[0]))
# Save the quartiles
citation_count_sum_25 = np.percentile(df_journals['citation_count_sum'], 25)
citation_count_sum_50 = np.percentile(df_journals['citation_count_sum'], 50)
citation_count_sum_75 = np.percentile(df_journals['citation_count_sum'], 75)

# Calculate the thresholds
IQR_citation_count_sum = citation_count_sum_75 - citation_count_sum_25
Lower_Limit = citation_count_sum_50 - IQR_citation_count_sum * 1.5
Upper_Limit = citation_count_sum_50 + IQR_citation_count_sum * 1.5

# Remove the outliers
df_journals = df_journals.loc[(df_journals['citation_count_sum'] > Lower_Limit) &
                              (df_journals['citation_count_sum'] < Upper_Limit)]
df_journals.boxplot(["citation_count_sum"])
print("Number of samples after: {}".format(df_journals.shape[0]))


# #### Learn
# Nice! You were successful in removing the outliers.

# ## Correlations: Multivariate statistics
# #### Try
# Use a correlation matrix for all features, plotting a scatter plot for each combination of numerical features in your dataset.

# In[24]:


sns.set(style="whitegrid", context="notebook")
sns.pairplot(df_journals[["citation_count_sum", "paper_count_sum",
                        "avg_cites_per_paper","proj_ai"]])


# #### Is it broken?
# There is one highly correlated feature.

# #### Learn
# There is a strong nonlinear correlation between proj_ai x avg_cites_per_paper.
# There is also some correlation between other variables here.
# But we need numbers to make a decision whether to remove some highly correlated feature.
# For this, a heat map with values is useful.

# >### Question 6
# >##### Try
# >Use the `corr` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html?highlight=corr#pandas.DataFrame.corr)) in the df_journal dataframe to print the correlations.

# In[25]:


# Enter your code here
corr = df_journals.corr()
corr


# ##### Learn
# Notice that the diagonal is always 1 (one), because it represents the variable against itself.
# 
# ##### Is it broken?
# The **proj_ai** and **avg_cites_per_paper** variables are highly correlated (over 99%). 
# 
# ##### Fix
# Let's try removing one of them. You will remove the **proj_ai** feature from the dataset.

# In[26]:


del df_journals["proj_ai"]


# #### Learn
# Now, use another great tool, a `heatmap` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html?highlight=corr#pandas.DataFrame.corr)), to confirm the correlation between the variables and the target, after removing the correlation. 

# In[27]:


corr = df_journals.corr()
ax = sns.heatmap(corr, annot=True,)


# ## Conclusion
# 
# In this exercise, you practiced different ways to know your data.
# 
# You explored your data over several dimensions and views to bring your raw data to a state of processed data, ready to be used for your model.
# 
# You needed to understand a short business scenario and corresponding dataset. You analyzed the data using descriptive statistics to better understand it.
# 
# You used visualization tools, including box and whisker plots and histograms, to support this analysis and understand the distribution of your data. You applied multivariate statistics with the help of scatter plots to spot correlations between features.
# 
# In **question 1**, you discovered the importance of a first glance at your data. You looked at how many features there were and how many samples. You looked at the first lines to see if anything caught your attention, such as missing values.
# 
# In **question 2**, you learned and practiced how to decide whether to remove or impute missing values. In the case of imputation, you practiced how to handle numerical or categorical imputation.
# 
# **Questions 3** gave you an understanding of basic statistics that you can apply to your data. You practiced plotting the distribution of features.
# 
# For **question 4**, you practiced with another important plot for numerical features: the box and whisker plot. This is a great tool for spotting outliers. 
# 
# **Question 5** showed you an interesting guideline for removing outliers using percentiles. You were challenged to implement the algorithm.
# 
# Finally, in **question 6** you learned and practiced using multivariate techniques, such as a correlation matrix and scatter plot, to spot correlations between features and how to deal with that.

# ### Good work!
# Now you have a dataset cleaned and ready to pass to the modeling step.
# 
