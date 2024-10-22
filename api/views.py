from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer, ensuring the files exist
model_path = './disease/model.pkl'
vectorizer_path = './disease/vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    raise FileNotFoundError("Model or vectorizer file not found.")

@api_view(['POST'])
def predict(request):
    symptoms = request.data.get('symptoms', '')

    # Check for symptoms before transforming
    if not symptoms:
        return Response({'error': 'No symptoms provided'}, status=400)

    transformed_symptoms = vectorizer.transform([symptoms])

    # Make prediction
    prediction = model.predict(transformed_symptoms)

    # Assuming prediction returns a single value, convert to a list for JSON response
    return Response({'prediction': prediction[0]})
