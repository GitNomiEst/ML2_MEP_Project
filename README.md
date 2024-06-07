# Machine Learning 2 - MEP Project

## For student
ATTENTION: There is another readme called 'READMEforSTUDENTS.md'. So for instructions on how to run the program etc. please read through the other README. Thanks!

## 1. What is the problem you are trying to solve? What is the motivation behind it? Why is your project relevant?
The problem revolves around assessing the potential threat posed by asteroids to Earth. Specifically, the challenge entails accurately predicting whether a given asteroid possesses characteristics that classify it as "dangerous" or not, in terms of its potential impact on our planet. This prediction involves a multi-faceted analysis of various parameters associated with the asteroid's magnitude, diameter, miss distance, and relative velocity.

The motivation behind this project is deeply rooted in the imperative to safeguard our planet and its inhabitants from potential cosmic hazards. While asteroid impacts are rare events on human timescales, they have the potential to cause catastrophic consequences, including widespread destruction, loss of life, and significant disruptions to ecosystems and global economies. Historic events, such as the Chelyabinsk meteor in 2013 and the extinction event that wiped out the dinosaurs, underscore the importance of proactive measures to mitigate the risks associated with near-Earth objects (NEOs).

## 2. Data Collection or Generation
The Data is collected directly from the NASA API / Asteroids - NeoWs / Neo - Feed (Link: https://api.nasa.gov/). It is therefore required to generate an API key and save it to a .env file where it can be collected from the code. Finally the data is saved to MongoDB, from where it can be read and used in the model. The MongoDB with the NEO-data can be accessed with user and password. These credentials can be provided by Noémie Käser on request.

## 3. Modeling
The model utilizes a Random Forest Classifier from SciKit Learn (Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). In this context, the model is trained to predict whether a given asteroid is potentially hazardous or not, based on various features extracted from near-Earth object (NEO) data.

To ensure the model is not biased towards the more frequent class, the dataset is balanced by ensuring there is an equal number of potentially hazardous and non-hazardous asteroids. The data is then split into a training set and a test set. The training set is used to train the model, while the test set is used to evaluate its performance. The model’s parameters are fine-tuned to find the best combination that gives the highest accuracy. Cross-validation is used to evaluate the model, ensuring it performs well on different subsets of the data.

The model is then trained on the training set using the best parameters found during tuning. Once trained, it is tested on the test set to measure its performance. Various metrics such as accuracy, precision, recall, F1 score, and AUC-ROC are calculated to evaluate the model. Additionally, various plots are generated to visualize the model’s performance, including the confusion matrix, precision-recall curve, feature importance, and ROC curve.

Once the model is trained, it can be used to predict whether a new asteroid is potentially hazardous based on its features. This model is particularly useful for identifying potentially hazardous asteroids, which allows scientists and researchers to monitor these asteroids, develop strategies to prevent or mitigate their impact, and allocate resources to the most dangerous ones.

(1) Model performance with all 5 attributes:
• Accuracy: 0.93
• Precision: 0.89
• Recall: 1.0
• F1 Score: 0.94
• AUC-ROC: 0.99
• AUC-PR: 0.99

(2) Model performance without miss distance:
• Accuracy: 0.93
• Precision: 0.91
• Recall: 0.98
• F1 Score: 0.94
• AUC-ROC: 0.98
• AUC-PR: 0.99

(3) Model performance without miss distance & relative velocity:
Accuracy: 0.93
Precision: 0.91
Recall: 0.98
F1 Score: 0.94
AUC-ROC: 0.99
AUC-PR: 0.99

All three models had high accuracy, precision, recall, F1 score, and AUC-PR, indicating strong performance. However, I chose the model number (3) with only three attributes, as the miss distance & relative velocity had a relatively low feature importance score.

After that I played around with the amount of data used in my model. As I was increasing this number I as well scored a better performance with only three attributes as it seemed to overfit, if I used all 5. As well, when predicting asteroid impacts, it is often more important to ensure that potentially dangerous events are not overlooked (high recall), even if this means that there may be some false-positive warnings. 
Final results of my chosen model:

Accuracy: 0.85
Precision: 0.79
Recall: 0.95
F1 Score: 0.86
AUC-ROC: 0.88
AUC-PR: 0.81


## 4. Interpretation and Validation 
Upon training the Random Forest Classifier model on NEO data, I proceeded to interpret and evaluate the performance of the model. The primary objective was to predict whether a given asteroid is potentially hazardous to Earth based on various features such as absolute magnitude, minimum diameter, maximum diameter, miss distance, and relative velocity. However, I chose the model number with only three attributes (without miss distance & relative velocity), as the two eliminated features had a relatively low feature importance score and the impact on the model was minor.

The model exhibited high performance, as evidenced by the evaluation metrics obtained during validation. Specifically, the model achieved a good Recall of 95%, as it is often more important to ensure that potentially dangerous events are not overlooked (high recall), even if this means that there may be some false-positive warnings. Additionally, the accuracy and precision scores further substantiate the model's effectiveness. The F1 score is 0.86, representing a balance between precision and recall and indicating as well the overall effectiveness of the model. The AUC-ROC is 0.88, measuring the model’s ability to distinguish between hazardous and non-hazardous asteroids, while the AUC-PR is 0.81, measuring the precision-recall trade-off, which is important for imbalanced datasets.

To further understand the factors influencing the model's predictions, I visualized the feature importance using a bar plot. This analysis revealed that the absolute magnitude, min and max diameter emerged as the most significant feature in determining their hazard level.

In terms of validation, the results were benchmarked against standard performance metrics for classification tasks. The achieved accuracy (0.85), precision (0.79), and recall (0.95) scores meet or exceed typical thresholds for satisfactory model performance, indicating the robustness and reliability of the approach. Additionally, the F1 score (0.86), AUC-ROC (0.88), and AUC-PR (0.81) further validate the model's effectiveness. The feature importance plot provided additional validation by clarifying the key drivers behind the model's predictions, thereby enhancing the interpretability and trustworthiness of the findings.
