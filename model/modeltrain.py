# This script should be executed separately from the Flask application to train the model

from model import load_neo_data, preprocess_data, train_model, evaluate_model, save_feature_importance_plot

if __name__ == "__main__":
    neo_data = load_neo_data()
    df = preprocess_data(neo_data)
    model, X_test, y_test = train_model(df)
    accuracy = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
    save_feature_importance_plot(model, df, 'frontend/static/feature_importance_plot.png')
