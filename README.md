# Data Mining and Machine Learning Project (F20DL - 2024/2025)

### Group Name: Dubai_UG 11

### Project Title: Medical Diagnosis of Lung Cancer

### Group Members:
1. Yasmeen Jasim	
2. Abdallah Moosa
3. Lourde Hajjar	
4. Lukas Ibne Jabbar
5. Ashar Ejaz

---

## Project Milestones and Timeline

| Milestone                  | Description                                               | Expected Completion |
|----------------------------|-----------------------------------------------------------|---------------------|
| Week 4 - Project Pitch     | Finalize topic, datasets, and objectives                  | 4/10/2024       |
| Week 6 - Data Exploration  | Complete EDA and preprocessing                            | 20/10/2024       |
| Week 8 - Initial Models    | Implement clustering and baseline models                  | 3/11/2024       |
| Week 10 - Advanced Models  | Train and evaluate advanced neural network models         | 15/11/2024          |
| Week 11 - Final Deliverables | Submit report, code, and complete project documentation | 22/11/2024          |

---

## Dataset(s) Sources

1. *Dataset1*  
   - Source: [(https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer)]
   - Type: Tabular
   - License: CC BY-NC-SA 4.0
   - Example: 

| GENDER | AGE | SMOKING | YELLOW_FINGERS | ANXIETY | PEER_PRESSURE | CHRONIC DISEASE | FATIGUE | ALLERGY | WHEEZING | ALCOHOL CONSUMING | COUGHING | SHORTNESS OF BREATH | SWALLOWING DIFFICULTY | CHEST PAIN | LUNG_CANCER |
|--------|-----|---------|----------------|---------|---------------|------------------|---------|---------|----------|--------------------|----------|---------------------|------------------------|------------|-------------|
| M      |  69 |       1 |              2 |       2 |             1 |                1 |       2 |       1 |        2 |                  2 |        2 |                   2 |                      2 |          2 | YES         |
| M      |  74 |       2 |              1 |       1 |             1 |                2 |       2 |       2 |        1 |                  1 |        1 |                   2 |                      2 |          2 | YES         |
| F      |  59 |       1 |              1 |       1 |             2 |                1 |       2 |       1 |        2 |                  1 |        2 |                   2 |                      1 |          2 | NO          |
| M      |  63 |       2 |              2 |       2 |             1 |                1 |       1 |       1 |        1 |                  2 |        1 |                   1 |                      2 |          2 | NO          |
| F      |  63 |       1 |              2 |       1 |             1 |                1 |       1 |       1 |        2 |                  1 |        2 |                   2 |                      1 |          1 | NO          |


2. *Dataset2* 
   - Type: Tabular 
   - Source: [https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link?resource=download]  
   - License: Other  
   - Example: 


| Index | Patient Id | Age | Gender | Air Pollution | Alcohol Use | Dust Allergy | Occupational Hazards | Genetic Risk | Chronic Lung Disease | Fatigue | Weight Loss | Shortness of Breath | Wheezing | Swallowing Difficulty | Clubbing of Finger Nails | Frequent Cold | Dry Cough | Snoring | Level   |
|-------|------------|-----|--------|---------------|-------------|--------------|-----------------------|--------------|-----------------------|---------|-------------|---------------------|----------|------------------------|--------------------------|---------------|-----------|---------|---------|
| 0     | P1         |  33 |      1 |             2 |           4 |            5 |                     4 |            3 |                     2 |       3 |           4 |                   2 |        2 |                      3 |                        1 |             2 |         3 |       4 | Low     |
| 1     | P10        |  17 |      1 |             3 |           1 |            5 |                     3 |            4 |                     2 |       1 |           3 |                   7 |        8 |                      6 |                        2 |             1 |         7 |       2 | Medium  |
| 2     | P100       |  35 |      1 |             4 |           5 |            6 |                     5 |            5 |                     4 |       8 |           7 |                   9 |        2 |                      1 |                        4 |             6 |         7 |       2 | High    |
| 3     | P1000      |  37 |      1 |             7 |           7 |            7 |                     7 |            6 |                     7 |       4 |           2 |                   3 |        1 |                      4 |                        5 |             6 |         7 |       5 | High    |
| 4     | P101       |  46 |      1 |             6 |           8 |            7 |                     7 |            7 |                     6 |       3 |           2 |                   4 |        1 |                      4 |                        2 |             4 |         2 |       3 | High    |




3. *Dataset3*  
   - Type: Images
   - Source: [https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset/data]  
   - License: Data files © Original Authors  
   - Example:





     **Bengin**:
     ![image](https://github.com/user-attachments/assets/4fd8e98d-239a-425a-a3b7-63a47cc07f47)


     **Malignant**:
     ![image](https://github.com/user-attachments/assets/f7c81986-663d-412a-9ff9-823542b2bd0c)


     **Normal**:
     ![image](https://github.com/user-attachments/assets/876b756c-f3be-4811-b13c-482fc071977e)
  





   

### Additional Steps in Dataset Preparation and the datasets


1. **Dataset 1**:
   - extracting features
   - balancing data using sampling techniques
   - normalizing the dataset (from 0 to 1)
   - convert to string


2. **Dataset 2**
   - normalizing the dataset
   - reducing the features (for DBSCAN)
   - convert to string


3. **Dataset 3**
   - edge detection applied on images
   - focused image


for every dataset, the files are formated and abstracted such that :
   - ef = extracted
   - b = balanced
   - nrml = normalized
   - og = original
   - str = string
   - ed = edge detection on image dataset
   - foc = focused image on dataset
   - ub = Unbalanced 


## Project Requirements Overview

This project satisfies the following requirements:

1. [*Data Analysis and Exploration*](https://github.com/machine-learning-hwu-project/Dubai_UG_11/tree/main/notebooks/r2-data-analysis) 
   - Preprocessing for missing values, outliers, etc.  
   - Exploratory Data Analysis (EDA) with visualizations.  
   - extracting Features for better performance.

2. [*Clustering*](https://github.com/machine-learning-hwu-project/Dubai_UG_11/tree/main/notebooks/r3-clustering)
   - Implementation of K-means, agglomerative-hierchal clustering and DBSCAN to identify patterns in data.
   - Exploring parameters such as number of clusters , min samples and epislon and type of dataset.
   - silhoutte score , Davies-Bouldin index were used to validate clusters.

3. [*Baseline Training and Evaluation*](https://github.com/machine-learning-hwu-project/Dubai_UG_11/tree/main/notebooks/r4-basic-classifiers-and-decision-trees)
   - Implementation of Decision Trees, Naive Bayes and K- nearest neighbor.
   - compare different datasets and parameters.

4. [*Neural Networks*](https://github.com/machine-learning-hwu-project/Dubai_UG_11/tree/main/notebooks/r5-neural%20networks)
   - Application of Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs) and logistic regression.
   - explore different datasets and parameters.

---


## Files/Folders in GitHub

| File/Folder     | Purpose                                      |
|------------------|----------------------------------------------|
| data/         | Contains raw and processed for datasets.        |
| notebooks/    | Contains jupyter notebooks for data analysis, clustering, classifiers, decision trees, neural networksand etc.  |
| scripts/      | Contains scripts used for data preprocessing.|
| README.md     | Project overview and instructions.          |

---

