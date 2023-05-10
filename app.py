from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model2.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    SepalLengthCm = request.form.get('SepalLengthCm')
    SepalWidthCm = request.form.get('SepalWidthCm')
    PetalLengthCm = request.form.get('PetalLengthCm')
    PetalWidthCm = request.form.get('PetalWidthCm')

    # result = {
    #     'SepalLengthCm':SepalLengthCm,
    #     'SepalWidthCm':SepalWidthCm,
    #     'PetalLengthCm':PetalLengthCm,
    #     'PetalWidthCm':PetalWidthCm
    # }
    #
    # return jsonify(result)

    input_query = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    result = model.predict(input_query)
    return jsonify({'Species': str(result)})


if __name__ == '__main__':
    app.run(debug=True)

# import pandas as pd
# import streamlit as st
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
#
# # Load Iris dataset
# iris = load_iris()
#
# # Create dataframe
# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# df['target'] = iris.target
#
# # Create a random forest classifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(df[iris.feature_names], df['target'])
#
#
# # Define the predict function
# def predict(sepal_length, sepal_width, petal_length, petal_width):
#     input_df = pd.DataFrame({
#         'sepal length (cm)': [sepal_length],
#         'sepal width (cm)': [sepal_width],
#         'petal length (cm)': [petal_length],
#         'petal width (cm)': [petal_width]
#     })
#     prediction = rf.predict(input_df)
#     return iris.target_names[prediction[0]]
#
#
# # Create a Streamlit app
# def main():
#     st.title('Iris Species Prediction API')
#
#     # Add input fields
#     sepal_length = st.number_input('Sepal length (cm)')
#     sepal_width = st.number_input('Sepal width (cm)')
#     petal_length = st.number_input('Petal length (cm)')
#     petal_width = st.number_input('Petal width (cm)')
#
#     # Make a prediction using the random forest model
#     if st.button('Predict'):
#         species = predict(sepal_length, sepal_width, petal_length, petal_width)
#         st.write('Predicted Species:', species)
#
#
# if __name__ == '__main__':
#     main()
