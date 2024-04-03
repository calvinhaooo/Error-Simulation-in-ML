# Error-Simulation-in-ML
Group5 project for data preparation
## The introduction of oue project
### Goal: 
create “bad” instances of data, which will help in evaluating the robustness of our model against realistic data issues. Create, test and evaluate models for robustness.
### Details: 
We aim to build an ML pipeline with the necessary preprocessing and cleaning operations, as well as various functions to inject non-malicious errors into the data. These errors will expose the weakness as well as help increase the robustness of the model.  
## The general process of our project 
*  Divide members into sub-groups for each dataset (Divide and conquer)
*  Apply preprocessing and data cleaning
*  Introduce errors into the datasets
*  Adapt preprocessing and cleaning operations
*  Merge sub-groups, compare results, refine successful adaptations 
   <div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/04545085-97d0-4b3c-acd4-5c8691534c94">
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


## Data Clean

## Experiments
* regression
For the regression task, the dataset we chose is `housing_price_dataset`. 

* classification
We chose dataset `renttherunway_final_data` as our classification dataset.
1. univariate outlier:
2. multivariate outlier: Function `generate_multivariate_outliers` can generate a set of multivariate outliers according to a given dataframe. We focus on two key parameters, 'percentage' and 'factors', and test their influence to model by changing their values. Next, function `merge_outliers` can merge outliers and original dataset and provides a visualization after dimensionality reduction.
3. missing: You can run 
4. label error:
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
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/8ece9d8a-639e-4ff3-9c38-44d336d9209d" width="800" height="200">
   <p>The result of labels error in ML pipeline training</p>
   </div>


