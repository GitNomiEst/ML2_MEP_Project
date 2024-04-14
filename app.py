from flask import Flask, jsonify, render_template, request
from model.model import load_neo_data, preprocess_data, train_model, evaluate_model, save_feature_importance_plot, predict_danger, balance_dataset, model

# Ausführen von Code aus model.py
neo_data = load_neo_data()  # Laden der NEO-Daten
df = preprocess_data(neo_data)  # Vorverarbeitung der Daten
balance_dataset (df)
model, X_test, y_test = train_model(df)  # Trainieren des Modells
accuracy = evaluate_model(model, X_test, y_test)  # Evaluieren der Modellgenauigkeit
save_feature_importance_plot(model, df, 'frontend/static/feature_importance_plot.png')  # Speichern des Feature-Importance-Plots

# Initialisierung der Flask-App
app = Flask(__name__, static_url_path='/', static_folder='frontend', template_folder='frontend/build')

# Routen für die verschiedenen Seiten der Webanwendung definieren

# Hauptseite
@app.route('/')
@app.route('/index.html')
def main_page():
    return render_template('index.html')

# Seite des Modells
@app.route('/model.html')
def model_page():
    return render_template('model.html')

# API-Endpunkt für die Vorhersage
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Daten aus der Anfrage erhalten

    # Formulardaten erhalten
    absolute_magnitude = float(data['absolute-magnitude'])
    min_diameter = float(data['min-diameter'])
    max_diameter = float(data['max-diameter'])
    miss_distance = float(data['miss-distance'])
    relative_velocity = float(data['relative-velocity'])

    # Gefahrenstufe vorhersagen
    danger_level = predict_danger(model, absolute_magnitude, min_diameter, max_diameter, miss_distance, relative_velocity)

    # Vorhersageergebnis zurückgeben
    if danger_level:
        prediction_message = "Your asteroid is potentially hazardous!"
    else:
        prediction_message = "Planet Earth is safe."

    return jsonify({'result': prediction_message})


if __name__ == "__main__":
    app.run(debug=True, port=80)
