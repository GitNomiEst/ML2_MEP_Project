from flask import Flask, jsonify, render_template, request
from model.model import load_neo_data, preprocess_data, train_model, evaluate_model, predict_danger, balance_dataset, use_trained_model, model

neo_data = load_neo_data()
df = preprocess_data(neo_data)
balanced_df = balance_dataset(df)
model, X_test, y_test = train_model(balanced_df)
accuracy, precision, recall, f1, auc_roc, auc_pr, cm = evaluate_model(model, X_test, y_test, balanced_df)

app = Flask(__name__, static_url_path='/', static_folder='frontend', template_folder='frontend/build')

# Main
@app.route('/')
@app.route('/index.html')
def main_page():
    return render_template('index.html')

# Model page
@app.route('/model.html')
def model_page():
    
    return render_template('model.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc_roc=auc_roc, auc_pr=auc_pr, cm=cm)

# API-Endpunkt für die Vorhersage
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Formulardaten erhalten
    absolute_magnitude = float(data['absolute-magnitude'])
    min_diameter = float(data['min-diameter'])
    max_diameter = float(data['max-diameter'])


    # Gefahrenstufe vorhersagen
    danger_level = predict_danger(model, absolute_magnitude, min_diameter, max_diameter)

    # Vorhersageergebnis zurückgeben
    if danger_level:
        prediction_message = "Your asteroid is potentially hazardous!"
    else:
        prediction_message = "Planet Earth is safe."

    return jsonify({'result': prediction_message})


if __name__ == "__main__":
    app.run(debug=True, port=80)
