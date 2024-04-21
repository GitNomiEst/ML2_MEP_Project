# Machine Learning 2 - MEP Project

## 1. What is the problem you are trying to solve? What is the motivation behind it? Why is your project relevant?
The problem at hand revolves around assessing the potential threat posed by asteroids to Earth. Specifically, the challenge entails accurately predicting whether a given asteroid possesses characteristics that classify it as "dangerous" or not, in terms of its potential impact on our planet. This prediction involves a multi-faceted analysis of various parameters associated with the asteroid's magnitude, diameter, miss distance, and relative velocity.

The motivation behind this project is deeply rooted in the imperative to safeguard our planet and its inhabitants from potential cosmic hazards. While asteroid impacts are rare events on human timescales, they have the potential to cause catastrophic consequences, including widespread destruction, loss of life, and significant disruptions to ecosystems and global economies. Historic events, such as the Chelyabinsk meteor in 2013 and the extinction event that wiped out the dinosaurs, underscore the importance of proactive measures to mitigate the risks associated with near-Earth objects (NEOs).

## 2. Data Collection or Generation
The Data is collected directly from the NASA API / Asteroids - NeoWs / Neo - Feed (Link: https://api.nasa.gov/). It is therefore required to generate an API key and save it to a .env file where it can be collected from the code. Finally the data is saved to MongoDB, from where it can be read and used in the model.

## 3. Modeling
The model utilizes a Random Forest Classifier. In this context, the model is trained to predict whether a given asteroid is potentially hazardous or not based on various features extracted from near-Earth object (NEO) data obtained from the NASA API.

The preprocessing stage involves extracting relevant features such as absolute magnitude, estimated diameter, miss distance, and relative velocity from the NEO data. These features serve as input variables for the model, enabling it to learn patterns and relationships indicative of the hazardous nature of asteroids.

To address potential class imbalance in the dataset, a function for balancing the dataset is provided, ensuring equal representation of hazardous and non-hazardous asteroids in the training data. This step enhances the model's ability to generalize well to new, unseen data.

The training phase involves splitting the preprocessed data into training and test sets, followed by training the Random Forest Classifier on the training data. Once trained, the model's performance is evaluated using metrics such as accuracy, precision, and recall on the test set, providing insights into its effectiveness in identifying hazardous asteroids.

Additionally, the model's feature importance is visualized through a bar plot, highlighting the relative significance of each input feature in determining the asteroid's hazard level. This visualization aids in understanding the factors driving the model's predictions and can inform further analysis or feature selection efforts.

Finally, the trained model is capable of making predictions on new asteroid data, allowing users to assess the potential danger posed by asteroids based on their characteristics. This predictive capability serves as a valuable tool for planetary defense efforts and contributes to the understanding of the risks associated with near-Earth objects.


## 4. Interpretation and Validation 
Upon training the Random Forest Classifier model on NEO data, I proceeded to interpret and evaluate the performance of the model. The primary objective was to predict whether a given asteroid is potentially hazardous to Earth based on various features such as absolute magnitude, estimated diameter, miss distance, and relative velocity.

The model exhibited high performance, as evidenced by the evaluation metrics obtained during validation. Specifically, the model achieved an impressive accuracy score of 98.60%, indicating its proficiency in correctly classifying hazardous and non-hazardous asteroids. Additionally, the precision and recall scores further substantiate the model's effectiveness, with precision at 86.21% and recall at 92.59%. These metrics signify the model's ability to accurately identify potentially hazardous asteroids while minimizing false positives and false negatives.

To further understand the factors influencing the model's predictions, we visualized the feature importance using a bar plot. This analysis revealed that the relative velocity of asteroids during close approach emerged as the most significant feature in determining their hazard level, followed by absolute magnitude and miss distance. This insight underscores the importance of velocity dynamics in assessing asteroid threat levels and aligns with established astronomical principles.

In terms of validation, the results were benchmarked against standard performance metrics for classification tasks. The achieved accuracy, precision, and recall scores surpass typical thresholds for satisfactory model performance, indicating the robustness and reliability of the approach. Moreover, the feature importance plot provided additional validation by elucidating the key drivers behind the model's predictions, thereby enhancing the interpretability and trustworthiness of the findings.


## Spec's (stable funtioning application):
- OS: running on Windows 11 Home
- Processor / RAM: Intel Core i7-1185G7, 16 GB RAM
- IDE used: Visual Studio Code (VSC) > developed by Microsoft

## Instructions:
Start app: Flask run
App starts on: http://localhost:5000/


## Supporting commands:
Update requirements: pip freeze > requirements.txt