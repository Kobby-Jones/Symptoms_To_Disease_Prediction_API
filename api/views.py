from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer





# from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# remove hashtags
def hashtags(text):
  hash = re.findall(r"#(\w+)", text)
  return hash

# translate emoji
# def emoji(text):
#   for emot in UNICODE_EMOJI:
#     if text == None:
#       text = text
#     else:
#       text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
#     return text

# remove retweet username and tweeted at @username
def remove_users(tweet):
  '''Takes a string and removes retweet and @user information'''
  tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
  # remove tweeted at
  return tweet

# remove links
def remove_links(tweet):
  '''Takes a string and removes web links from it'''
  tweet = re.sub(r'http\S+', '', tweet) # remove http links
  tweet = re.sub(r'bit.ly/\S+', '', tweet) # remove bitly links
  tweet = tweet.strip('[link]') # remove [links]
  return tweet
def clean_html(text):
  html = re.compile('<.*?>')#regex
  return html.sub(r'',text)

# remove non ascii character
def non_ascii(s):
  return "".join(i for i in s if ord(i)<128)

def lower(text):
  return text.lower()

# remove stopwords
def removeStopWords(str):
#select english stopwords
  cachedStopWords = set(stopwords.words("english"))
#add custom words
  cachedStopWords.update(('and','I','A','http','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','mailto','regards','ayanna','like','email'))
#remove stop words
  new_str = ' '.join([word for word in str.split() if word not in cachedStopWords]) 
  return new_str

# remove email address
def email_address(text):
    if not isinstance(text, str):
        return ''  # Or handle it as needed
    email = re.compile(r'[\w\.-]+@[\w\.-]+')
    return email.sub(r'', text)


def punct(text):
  token=RegexpTokenizer(r'\w+')#regex
  text = token.tokenize(text)
  text= " ".join(text)
  return text 

# remove digits and special characters
def remove_digits(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    #when the ^ is on the inside of []; we are matching any character that is not included in this expression within the []
    return re.sub(pattern, '', text)

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

def remove_(tweet):
  tweet = re.sub('([_]+)', "", tweet)
  return tweet


def clean_text(text):
   text = remove_users(text)
   text = remove_links(text)
   text = remove_digits(text)
   text = remove_special_characters(text)
   text = removeStopWords(text)
   text = remove_(text)
   text = punct(text)
   text = clean_html(text)
   text = non_ascii(text)
   text = hashtags(text)
   text = email_address(text)
   text = lower(text)
   return text




# Load your trained model and vectorizer, ensuring the files exist
model_path = './disease/Disease_model.pkl'
vectorizer_path = './disease/Vectorizers.pkl'
label_encoder_path = './disease/label_encoder.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)


    
else:
    raise FileNotFoundError("Model or vectorizer file not found.")

@api_view(['POST'])
def predict(request):
    symptoms = request.data.get('symptoms', '')

    # Check for symptoms before transforming
    if not symptoms:
        return Response({'error': 'No symptoms provided'}, status=400)

    # Check if symptoms is a string
    if not isinstance(symptoms, str):
        return Response({'error': 'Symptoms must be a string'}, status=400)

    cleaned_text = clean_text(symptoms)
    emb_text = vectorizer.transform([cleaned_text])  # Transform the cleaned text
    prediction = model.predict(emb_text)  # Predict the disease
    label = label_encoder.inverse_transform(prediction)  # Get the label in text form

    # Assuming prediction returns a single value, convert to a list for JSON response
    return Response({'prediction': label[0]})

