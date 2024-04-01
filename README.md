# Error-Simulation-in-ML
Group5 project for data preparation
## The general process of our project 
![image](https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/968f96a2-f8e9-4fd1-a071-318abfe7beda)

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


* classification
1. outlier:
2. duplicate: 
3. missing: 
4. label error:
   For label errors, we can use `modify_labels_to_negative(labels, percentage)` in `error_genertor.py` to simulate these errors. The experiment is based on IMDb dataset, we wanna see how label errors can influence the ML pipeline training. The parameter percentage we set is 25 and 50, we gonna change 25% or 50% negative labels into positive to simulate these error. Then, we can use `clean_bc_label_error(train_data, train_labels)` to detect these errors in `data_cleaner.py`. Then we can get the data is pruned, and substitute it as the correct label.

<center class="half">
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/def394a3-b256-4994-8a13-f28ca7031341" width="400" height="200">
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/8382f130-93fd-4326-931a-bd2a4e3af27b" width="400" height="200">
   <p>Before Modification                                                   After Modification</p>
</center>
<center>
   <img src="https://github.com/calvinhaooo/Error-Simulation-in-ML/assets/145265103/e6ce9689-1c1c-40d4-b8d0-af922c4f2586" width="400" height="200">
   <p>After cleaned</p>
</center>










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
