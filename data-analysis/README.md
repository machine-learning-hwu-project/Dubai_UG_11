# Data Analysis Section
This section focuses on the data analysis part of the project, where we analyze and pre-process the dataset before applying any crucial algorithms (R3-R5) :

## Dataset Overview
The initial analyzing of the dataset revealed an imbalance in target classes. This imbalance could affect model performance by biasing predictions toward the majority class which is the Not lung cancer class. Below is a snapshot of the dataset’s class distribution:

(paste picture of unbalanced dataset)

## Oversampling Technique

To analyze and test, oversampling was used to address the imbalance of the original dataset where originally, the output class "YES" included 270 instances and the output class "NO" included 39 instances (minority class). Random oversampling was applied to improve the models performance on the minority class and prevent bias toward the majority class. f1 score, recall and precision were measured for the balanced class. 
[`code`](/data-analysis/oversampling_31_10.ipynb/)


![image](https://github.com/user-attachments/assets/81386b19-4de1-44f8-8ae0-289f3da44f61)




## Model Testing with Balanced and Imbalanced Datasets

To analyze how balancing affects on model performance, tests were conducted on both the imbalanced and balanced versions of the dataset. The following notebooks document this process:

[`unbalanced_tabular_early testing`](/data-analysis/unbalanced_tabular_early_testing.ipynb/) : Tests conducted using the original, imbalanced dataset.

[`balanced_tabular_early testing`](/data-analysis/balanced_tabular_early_testing.ipynb/) : Tests conducted using the oversampled, balanced dataset.


Below are the confusion matrices from each notebook, showing the difference in model performance:






### balanced Dataset Confusion Matrix:

![image](https://github.com/user-attachments/assets/c2b111d3-d414-4ff1-adb2-c0297d5a04fa)



### imbalanced Dataset Confusion Matrix:

![image](https://github.com/user-attachments/assets/2e0c2700-9831-452f-90ec-cc7a79dedaf8)



## Conclusion and steps forward:

