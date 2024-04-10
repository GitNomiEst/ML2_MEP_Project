import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
    client = MongoClient(f"mongodb+srv://nomi:012no.AhM@nasa.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
    db = client['nasa']
    collection = db['nasa']

    # Fetch data from MongoDB
    neo_data = list(collection.find())
    print("data from mongo loaded")

    return neo_data


def preprocess_data(neo_data):
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
    print("data preprocessed")

    return df

def train_model(dataframe):
    # Train a random forest classifier model using the provided DataFrame.
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataframe.drop('is_potentially_hazardous', axis=1), 
                                                        dataframe['is_potentially_hazardous'], test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("model trained")
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Nachdem das Modell trainiert wurde
    predictions = model.predict(X_test)

    # Model evaluation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print("Precision:", precision)
    print("Recall:", recall)

    return accuracy

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
    plt.savefig(plot_filename)
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
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    #print("Accuracy:", accuracy)
    save_feature_importance_plot(model, df, 'frontend/static/feature_importance_plot.png')

