import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,  precision_recall_curve, roc_curve, f1_score, roc_auc_score, auc, roc_curve

#COMMENTED OUT AS DATA SHALL BE LOADED FROM MONGO DB TO NOT EXCEED THE NASA API LIMIT
#def load_neo_data(api_key):
    #print("Start Data load from API...")
    #neo_data = get_neo_data(api_key)
    #print("Data loaded from API")
    #return neo_data

# RandomForestClassifier
model = RandomForestClassifier()


def load_neo_data():
    # Connect to MongoDB
    client = MongoClient(f"mongodb+srv://kaeseno1:PW@cluster0.4pnoho7.mongodb.net/")
    db = client['nasa']
    collection = db['nasa']

    # Fetch data from MongoDB
    neo_data = list(collection.find())
    print("\nProgress / Completed tasks:")
    print("Data loaded from DB")

    return neo_data


def preprocess_data(neo_data):
    #print (neo_data)
    
    features = []
    labels = []

    for data in neo_data:
        near_earth_objects = data.get('near_earth_objects', {})
        for date, asteroids in near_earth_objects.items():
            for asteroid in asteroids:
                features.append([
                    asteroid['absolute_magnitude_h'],
                    asteroid['estimated_diameter']['kilometers']['estimated_diameter_min'],
                    asteroid['estimated_diameter']['kilometers']['estimated_diameter_max'],
                    asteroid['close_approach_data'][0]['miss_distance']['kilometers'],
                    asteroid['close_approach_data'][0]['relative_velocity']['kilometers_per_hour']
                ])
                labels.append(asteroid['is_potentially_hazardous_asteroid'])

    # Create a DataFrame from the extracted features and labels
    df = pd.DataFrame(features, columns=['absolute_magnitude_h', 'min_diameter_km', 'max_diameter_km', 'miss_distance_km', 'relative_velocity_km_hour'])
    df['is_potentially_hazardous'] = labels
    print("Data preprocessed")

    return df

def balance_dataset(df):
    
    # Separate hazardous and non-hazardous asteroids
    hazardous_asteroids = df[df['is_potentially_hazardous'] == True]
    non_hazardous_asteroids = df[df['is_potentially_hazardous'] == False]
    
    # Determine the number of hazardous and non-hazardous asteroids
    num_hazardous = len(hazardous_asteroids)
    num_non_hazardous = len(non_hazardous_asteroids)
    
    if num_hazardous < num_non_hazardous: #undersampling
        sampled_non_hazardous = non_hazardous_asteroids.sample(n=num_hazardous, random_state=42)
        balanced_df = pd.concat([sampled_non_hazardous, hazardous_asteroids])
         
    if num_hazardous > num_non_hazardous: 
        sampled_hazardous = hazardous_asteroids.sample(n=num_non_hazardous, replace=True, random_state=42)
        balanced_df = pd.concat([non_hazardous_asteroids, sampled_hazardous])   
    
    # Shuffle dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Dataset balanced")

  
    return balanced_df

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f'Best parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_



def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f'Cross-validation accuracy scores: {scores}')
    print(f'Mean accuracy: {np.mean(scores)}, Std: {np.std(scores)}')
    return scores


def train_model(dataframe):
    print("starting training")
    X = dataframe.drop('is_potentially_hazardous', axis=1)
    y = dataframe['is_potentially_hazardous']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Tuning hyperparams")
    
    # Tune hyperparameters
    model = tune_hyperparameters(X_train, y_train)
    
    print("Cross validate model")
    
    # Cross-validate the model
    cross_validate_model(model, X, y)
    
    # Train the model with the best parameters
    model.fit(X_train, y_train)
    print("Model trained")
    
    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, df):
    predictions = model.predict(X_test)

    # Model evaluation & round numbers
    accuracy = round(accuracy_score(y_test, predictions), 2)
    precision = round(precision_score(y_test, predictions), 2)
    recall = round(recall_score(y_test, predictions), 2)
    
    # Compute F1 score
    f1 = round(f1_score(y_test, predictions), 2)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig('frontend/static/confusion_matrix')
    print("Saved Confusion Matrix")
    
    # Clear any existing figures
    plt.clf()


    # Generate probability predictions
    probabilities = model.predict_proba(X_test)[:, 1]

    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    pr_auc = round(auc(recall_curve, precision_curve), 2)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc})')
    plt.savefig('frontend/static/precision_recall_curve.png')
    print("Saved Precision-Recall Curve plot")

    
    plt.clf()
    
    # Feature importance
    feature_importances = model.feature_importances_
    plt.bar(df.columns[:-1], feature_importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('frontend/static/feature_importance_plot.png')
    print("Saved Feature Importance plot")
    
    plt.clf()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = round(auc(fpr, tpr), 2)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc})')
    plt.savefig('frontend/static/roc_curve.png')
    print("Saved ROC Curve plot")
    
    print("\nMetrics:")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC-ROC:", roc_auc)
    print("AUC-PR:", pr_auc)

    return accuracy, precision, recall, f1, roc_auc, pr_auc, cm


def predict_danger(model, absolute_magnitude, min_diameter, max_diameter, miss_distance, relative_velocity):
    # Preprocess input data
    input_data = pd.DataFrame([[absolute_magnitude, min_diameter, max_diameter, miss_distance, relative_velocity]],
                              columns=['absolute_magnitude_h', 'min_diameter_km', 'max_diameter_km', 'miss_distance_km', 'relative_velocity_km_hour'])

    # Make prediction using the trained model
    prediction = model.predict(input_data)

    # Return the predicted danger level
    return prediction[0]

if __name__ == "__main__":
    neo_data = load_neo_data()
    df = preprocess_data(neo_data)
    balanced_dataset = balance_dataset (df)
    
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test, balanced_dataset)    
