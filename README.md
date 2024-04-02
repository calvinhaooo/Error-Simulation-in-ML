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
* **RentRunAway**: About Clothing fit data with 192,544 rorws and 15 columns
* **IMDb dataset**: About IMDb movie review with 50,000 rows and 2 columns.
* **Housing price dataset**: Used for regression task for predicting the housing price with 50,000 rows and 6 columns
* **SunSpot dataset**: a kind of time series dataset with 3265 rows and 2 columns.

## Processing

## Generating Errors
### Type of Errors:
1. noise
2. outlier
3. text
4. duplicate
5. label error
6. missing value


## Clean Data

## Experiments
* regression
For the regression task, the dataset we chose is `housing_price` dataset. 

* classification
1. outlier:
2. duplicate: 
3. missing: 
4. label error:
   For label errors, we can use `modify_labels_to_negative(labels, percentage)` in `error_genertor.py` to simulate these errors. The experiment is based on IMDb dataset, we wanna see how label errors can influence the ML pipeline training. The parameter percentage we set is 25 and 50, we gonna change 25% or 50% negative labels into positive to simulate these error. Then, we can use `clean_bc_label_error(train_data, train_labels)` to detect these errors in `data_cleaner.py`. Then we can get the data is pruned, and substitute it as the correct label. Below is the bar charts of simulating label errors and cleaning it. The left bar is the number of positive, the other one is Negative.

<center class="half">
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/def394a3-b256-4994-8a13-f28ca7031341" width="400" height="200", alt= "Before modification">
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/8382f130-93fd-4326-931a-bd2a4e3af27b" width="400" height="200">
   <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Before&nbsp;Modification&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After&nbsp;Modification</p>
</center>

<div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/e6ce9689-1c1c-40d4-b8d0-af922c4f2586" width="400" height="200">
   <p>After cleaned</p>
</div>
   After using `clean_bc_label_error` function to detect the error, we will get the confidence of every errors. According to this, we can correct it. The result is below:
   <p>&nbsp;</p>
   <div align=center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/f5706c61-cf22-4bdc-81c1-970e1af3b196" width="800" height="200">
   <p>The result of labels error in ML pipeline training</p>
   </div>











[//]: # (# without bra size)

[//]: # (# AUC Score: 0.8514183711474294)

[//]: # (# Accuracy: 0.7881227981882235)

[//]: # (# Precision: 0.7380110362275406)

[//]: # (# Recall: 0.6226882933203429)

[//]: # (# F1 Score: 0.6628743242436715)

[//]: # (# Confusion Matrix:)

[//]: # (# [[8924  321  288])

[//]: # (#  [1105  968  135])

[//]: # (#  [ 977  121 1070]])

[//]: # ()
[//]: # (# cleaned data)

[//]: # (# AUC Score: 0.8518708069803583)

[//]: # (# Accuracy: 0.7859302995391705)

[//]: # (# Precision: 0.7237415956573389)

[//]: # (# Recall: 0.6371173298094482)

[//]: # (# F1 Score: 0.6692058156810673)

[//]: # (# Confusion Matrix:)

[//]: # (# [[8731  350  392])

[//]: # (#  [1044  990  181])

[//]: # (#  [ 882  124 1194]])

[//]: # ()
[//]: # (# add outlier bra size)

[//]: # (# AUC Score: 0.8511242361643943)

[//]: # (# Accuracy: 0.7862183179723502)

[//]: # (# Precision: 0.7265108824576934)

[//]: # (# Recall: 0.6347286331629108)

[//]: # (# F1 Score: 0.6677928386486712)

[//]: # (# Confusion Matrix:)

[//]: # (# [[8757  323  393])

[//]: # (#  [1074  961  180])

[//]: # (#  [ 880  119 1201]])

[//]: # ()
[//]: # (# log, only text)

[//]: # (# AUC Score: 0.8298759894064216)

[//]: # (# Accuracy: 0.7758496023138105)

[//]: # (# Precision: 0.6936738196258047)

[//]: # (# Recall: 0.6172401446556065)

[//]: # (# F1 Score: 0.6465030296115503)

[//]: # (# Confusion Matrix:)

[//]: # (# [[9608  490  431])

[//]: # (#  [1164 1052  195])

[//]: # (#  [ 965  165 1143]])

[//]: # ()
[//]: # (# AUC Score: 0.8460666033583649)

[//]: # (# Accuracy: 0.7785858294930875)

[//]: # (# Precision: 0.7076563476990462)

[//]: # (# Recall: 0.6382360424106194)

[//]: # (# F1 Score: 0.6658485754518694)

[//]: # (# Confusion Matrix:)

[//]: # (# [[8588  473  412])

[//]: # (#  [ 998 1050  167])

[//]: # (#  [ 890  135 1175]])
