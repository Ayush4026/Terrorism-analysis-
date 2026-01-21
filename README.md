This project focuses on analyzing real-world terrorism incident data to predict the success of terrorist attacks and uncover hidden patterns using machine learning techniques. By applying both supervised and unsupervised learning models, the study aims to support data-driven insights for security analysis and preventive strategy formulation.

The dataset consists of 4,960 terrorism incidents with features related to time, location, attack type, target type, weapon type, casualties, and suicide involvement. Careful preprocessing techniques were applied, including missing value handling, feature selection, and Min–Max normalization to ensure reliable model performance.

For supervised learning, K-Nearest Neighbors (KNN), Logistic Regression, and Naive Bayes classifiers were implemented to predict whether an attack would be successful. Among these, KNN and Logistic Regression achieved the highest accuracy of approximately 86%, demonstrating strong predictive capability. Model evaluation was performed using accuracy scores and confusion matrices.

In addition, K-Means clustering was used as an unsupervised learning approach to identify underlying attack patterns. The Elbow Method was applied to determine the optimal number of clusters, helping reveal behavioral similarities among different types of attacks.

Key Highlights

1.Real-world terrorism dataset analysis

2.Data preprocessing with meaningful handling of casualties

3.Min–Max scaling for distance-based models

4.Supervised ML: KNN, Logistic Regression, Naive Bayes

5.Unsupervised ML: K-Means clustering with Elbow Method

6.Performance evaluation using accuracy and confusion matrix

7.Actionable insights for security and policy analysis

Tools & Technologies

1.Python

2.Pandas, NumPy

3.Matplotlib, Seaborn

4.Scikit-learn
