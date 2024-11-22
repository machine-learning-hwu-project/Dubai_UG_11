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
| Week 4 - Project Pitch     | Finalize topic, datasets, and objectives                  | [Insert Date]       |
| Week 6 - Data Exploration  | Complete EDA and preprocessing                            | [Insert Date]       |
| Week 8 - Initial Models    | Implement clustering and baseline models                  | [Insert Date]       |
| Week 10 - Advanced Models  | Train and evaluate advanced neural network models         | 15/11/2024          |
| Week 11 - Final Deliverables | Submit report, code, and complete project documentation | 22/11/2024          |

---

## Dataset(s) Sources

1. *Lung Cancer*  
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


2. *Lung Cancer Prediction* 
   - Type: Tabular 
   - Source: [https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link?resource=download]  
   - License: Other  
   - Example: 

3. *IQ-OTH/NCCD - Lung Cancer Dataset*  
   - Type: Images
   - Source: [https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset/data]  
   - License: Data files © Original Authors  
   - Example:   

| Index | Patient Id | Age | Gender | Air Pollution | Alcohol Use | Dust Allergy | Occupational Hazards | Genetic Risk | Chronic Lung Disease | Fatigue | Weight Loss | Shortness of Breath | Wheezing | Swallowing Difficulty | Clubbing of Finger Nails | Frequent Cold | Dry Cough | Snoring | Level   |
|-------|------------|-----|--------|---------------|-------------|--------------|-----------------------|--------------|-----------------------|---------|-------------|---------------------|----------|------------------------|--------------------------|---------------|-----------|---------|---------|
| 0     | P1         |  33 |      1 |             2 |           4 |            5 |                     4 |            3 |                     2 |       3 |           4 |                   2 |        2 |                      3 |                        1 |             2 |         3 |       4 | Low     |
| 1     | P10        |  17 |      1 |             3 |           1 |            5 |                     3 |            4 |                     2 |       1 |           3 |                   7 |        8 |                      6 |                        2 |             1 |         7 |       2 | Medium  |
| 2     | P100       |  35 |      1 |             4 |           5 |            6 |                     5 |            5 |                     4 |       8 |           7 |                   9 |        2 |                      1 |                        4 |             6 |         7 |       2 | High    |
| 3     | P1000      |  37 |      1 |             7 |           7 |            7 |                     7 |            6 |                     7 |       4 |           2 |                   3 |        1 |                      4 |                        5 |             6 |         7 |       5 | High    |
| 4     | P101       |  46 |      1 |             6 |           8 |            7 |                     7 |            7 |                     6 |       3 |           2 |                   4 |        1 |                      4 |                        2 |             4 |         2 |       3 | High    |

   

### Additional Steps in Dataset Preparation
- [List any additional data cleaning, augmentation, or collection steps performed.]

---

## Project Requirements Overview

This project satisfies the following requirements:

1. *Data Analysis and Exploration*  
   - Preprocessing for missing values, outliers, etc.  
   - Exploratory Data Analysis (EDA) with visualizations.  
   - Feature selection for optimal performance.

2. *Clustering*  
   - Implementation of [insert algorithm(s), e.g., "k-Means"] to identify patterns in data.  
   - Evaluation metrics: [Insert metrics used, e.g., "Silhouette Score"].

3. *Baseline Training and Evaluation*  
   - Implementation of Decision Trees and other baseline models like Naive Bayes.
   - Evaluation using [insert metrics, e.g., "Accuracy, RMSE"].

4. *Neural Networks*  
   - Application of Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs).  
   - Fine-tuning and comparison with baseline models.  

---

## Running the Data Preparation Pipeline


## Files/Folders in GitHub

| File/Folder     | Purpose                                      |
|------------------|----------------------------------------------|
| data/         | Contains raw and processed datasets.        |
| src/          | Source code for data processing and modeling.|
| notebooks/    | Jupyter notebooks for exploration and EDA.  |
| results/      | Model outputs, evaluation metrics, and figures.|
| README.md     | Project overview and instructions.          |

---

### Disclaimer:
This project adheres to the guidelines outlined in the coursework requirements. All work is original, and external contributions (e.g., datasets, libraries) are properly cited. 

