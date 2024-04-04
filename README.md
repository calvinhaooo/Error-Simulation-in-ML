# Error-Simulation-in-ML
Group5 project for data preparation
## The introduction of oue project
### Goal: 
create “bad” instances of data, which will help in evaluating the robustness of our model against realistic data issues. Build,test and evaluate different models for robustness.
### Details: 
We aim to build an ML pipeline with the necessary preprocessing and cleaning operations, as well as various functions to inject non-malicious errors into the data. These errors will expose the weakness as well as help increase the robustness of the model.  
## The general process of our project 
*  Divide members into sub-groups for each dataset (Divide and conquer)
*  Apply preprocessing and data cleaning
*  Introduce errors into the datasets
*  Adapt preprocessing and cleaning operations
*  Merge sub-groups, compare results, refine successful adaptations
Below is our general workflow of our project.
   <div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/5259a934-00c1-45d1-b940-b90e332d18f1">
   <p>The workflow of our project</p>
   </div>


## The dataset we used
* **Amazon product review**: The dataset demonstrates the product details of amazon website including the reviews of product from customer with 1,292,954 rows and 12 columns.
* **Molcloth dataset**: About Clothing fit with 82,790 rows and 15 columns
* **RentRunAway**: About Clothing fit data with 192,544 rows and 15 columns. Click [here](https://datarepo.eng.ucsd.edu/mcauley_group/data/renttherunway/renttherunway_final_data.json.gz) to download the dataset.
* **IMDb dataset**: About IMDb movie review with 50,000 rows and 2 columns. Click [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) to download the dataset.
* **Housing price dataset**: Used for regression task for predicting the housing price with 50,000 rows and 6 columns
* **SunSpot dataset**: a kind of time series dataset with 3265 rows and 2 columns.

## PreProcessing
We use `data_preprocessor.py` to preprocess datasets.

## Generating Errors
We use `error_generator.py` to simulate various error types including:
1. missing value
2. outlier
3. noise
4. label error
5. spelling error


## Data Clean

## Experiments
We tried several models to assess the impact of errors on different models and selected the one that best suited our needs. For instance, complex models like RandomForest are sometimes less affected by certain types of errors and may maintain good performance despite them. 
* regression
For the regression task, the dataset we chose is `housing_price_dataset` and the time series data. 

* classification
We chose dataset `renttherunway_final_data` as our classification dataset.
### Result 
1. univariate outlier: Function `add_univariate_outliers` in `error_generator.py` can generate univariate outlier according to a given data frame. We focus on two key parameters: `outlier_percentage` and `factor`. The first one is how many outliers u wanna inject into the dataset, the second one is to control how many times do you zoom in and out. Then, after injecting these errors, we can use the IQR method, the data is divided into quartiles and data points outside the range $[Q1-1.5\times IQR, Q3+1.5\times IQR]$ are labeled as outliers, where $IQR=Q3-Q1$. This function is in `data_cleaner.py` named `remove_outliers_iqr`. After running this, we can successfully detect and clean up most of these errors. The result as below shows:
<div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/1a3715bf-74c4-45f2-b230-362eca432904" width="600" height="150">
   <p>The result of univariate errors</p>
</div>

2. multivariate outlier: Function `generate_multivariate_outliers` can generate a set of multivariate outliers according to a given dataframe. We focus on two key parameters, 'percentage' and 'factors', and test their influence to model by changing their values. Next, function `merge_outliers` can merge outliers and original dataset and provides a visualization after dimensionality reduction.
<div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/ed14183a-f72c-428f-ac7c-5531bc58f933" width="700" height="150">
   <p>The result of multivariate errors</p>
</div>

3. missing: You can run function `add_null_noise` in the `error_generator.py`. The only parameter we focus on is the null_percentage. We can set different percentage to research how it affects the pipleline training. We choose 2 methods to clean the null noise: the first one is to throw it away, and the other is use KNN to impute these errors. In addition, we also change different vectorize in the pipeline.
<div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/4c6a01f7-0136-4e6a-8a24-a08484ce12c2" width="600" height="150">
   <p>The result of missing value errors</p>
</div>

4. Label error:
   For label errors, we can use `modify_labels_to_negative(labels, percentage)` in `error_genertor.py` to simulate these errors. The experiment is based on IMDb dataset, we wanna see how label errors can influence the ML pipeline training. The parameter percentage we set is 25 and 50, we gonna change 25% or 50% negative labels into positive to simulate these error. Then, we can use `clean_bc_label_error(train_data, train_labels)` to detect these errors in `data_cleaner.py`. Then we can get the data is pruned, and substitute it as the correct label. Below is the bar charts of simulating label errors and cleaning it. The left bar is the number of positive, the other one is Negative.

<center class="half">
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/def394a3-b256-4994-8a13-f28ca7031341" width="400" height="200", alt= "Before modification">
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/8382f130-93fd-4326-931a-bd2a4e3af27b" width="400" height="200">
   <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Before&nbsp;Modification&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After&nbsp;Modification</p>
</center>

<div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/e6ce9689-1c1c-40d4-b8d0-af922c4f2586" width="400" height="200">
   <p>After cleaned</p>
</div>
   After using `clean_bc_label_error` function to detect the error, we will get the confidence of every error. According to this, we can correct it. The result is below:
   <p>&nbsp;</p>
   <div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/8ece9d8a-639e-4ff3-9c38-44d336d9209d" width="800" height="150">
   <p>The result of labels error in ML pipeline training</p>
   </div>

Then, we have tried 3 unsupervised methods for detecting and cleaning up labels. The unsupervised methods we used are:
* nltk VADER
* LatentDirichletAllocation (LDA)
* KMeans clustering
After done the experiment of these methods in the `reviews_analyzer.ipynb`. After running this, the cleaning label we get is stored in the `predicted_reviews.csv`. Then we can use the label we cleaned for the machine learning pipeline, and we get the result as the table shown below:
<div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/133a11ac-fd7d-4e0c-945b-b685511cb3bc" width="800" height="150">
   <p>Unsupervised methods for label errors</p>
</div>

5. spelling error
In terms of spelling error, the function `random_replace_column` in the `error_generator.py` can be used to simulate these errors. For instance, now the label is 'fit', 'large', 'small'. We can add random character into it, then it will become like 'fit' 'fat' 'small' 'smaal' 'laage'. Then, we can use the function `random_replace_column` in `data_cleaner.py` to detect and clean these errors, the basic idea of it is KNN too.
<div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/35b5a4f1-ad27-4aef-99b7-c8aff22926d9" width="800" height="150">
   <p>Unsupervised methods for spelling errors</p>
</div>

6. noise

Simulate noise that occurs in time-series data, because of the natural characteristics of time-series, we can remove the noise  using filtering techniques.
![image](https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145342600/4e7e5a92-e6a2-4a1b-819b-49acfd843fb1)



