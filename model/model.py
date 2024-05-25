import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,  precision_recall_curve

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
    client = MongoClient(f"mongodb://localhost:27017/")
    db = client['nasa']
    collection = db['nasa']

    # Fetch data from MongoDB
    neo_data = list(collection.find())
    print("Data from mongo loaded")

    return neo_data


def preprocess_data(neo_data):
    print (neo_data)
    
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

# Usage of balance dataset:
# After preprocessing data, call balance_dataset function to balance the dataset balanced_df = balance_dataset(df)
# Then continue with training the model using the balanced dataset model, X_test, y_test = train_model(balanced_df)

def balance_dataset(df):
    
    # Separate hazardous and non-hazardous asteroids
    hazardous_asteroids = df[df['is_potentially_hazardous'] == True]
    non_hazardous_asteroids = df[df['is_potentially_hazardous'] == False]
    
    # Determine the number of hazardous and non-hazardous asteroids
    num_hazardous = len(hazardous_asteroids)
    num_non_hazardous = len(non_hazardous_asteroids)
    
    # Undersampling or oversampling
    if num_hazardous < num_non_hazardous:  # Undersampling
        # Sample non-hazardous asteroids equal to the number of hazardous ones
        sampled_non_hazardous = non_hazardous_asteroids.sample(n=num_hazardous, random_state=42)
        # Combine sampled non-hazardous with all hazardous asteroids
        balanced_df = pd.concat([sampled_non_hazardous, hazardous_asteroids])
    else:  # Oversampling
        # Sample hazardous asteroids with replacement equal to the number of non-hazardous ones
        sampled_hazardous = hazardous_asteroids.sample(n=num_non_hazardous, replace=True, random_state=42)
        # Combine sampled hazardous with all non-hazardous asteroids
        balanced_df = pd.concat([non_hazardous_asteroids, sampled_hazardous])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df



def train_model(dataframe):
    # Train a random forest classifier model using the provided DataFrame.
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop('is_potentially_hazardous', axis=1), 
                                                        dataframe['is_potentially_hazardous'], test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model trained")
    return model, X_test, y_test



def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    # Model evaluation & round numbers
    accuracy = round(accuracy_score(y_test, predictions), 2)
    precision = round(precision_score(y_test, predictions), 2)
    recall = round(recall_score(y_test, predictions), 2)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig('frontend/static/confusion_matrix')
    print("Saved Confusion Matrix")

  # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('frontend/static/precision_recall_curve')
    print("Saved Precision-Recall Curve plot")
    
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    


    return accuracy, precision, recall, cm

def save_feature_importance_plot(model, dataframe, plot_filename):
    # Plot feature importance
    feature_importances = model.feature_importances_
    plt.bar(dataframe.columns[:-1], feature_importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig('frontend/static/feature_importance_plot.png')
    print("Saved Feature Importance plot")


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
    balance_dataset (df)
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    save_feature_importance_plot(model, df, 'frontend/static/feature_importance_plot.png')

