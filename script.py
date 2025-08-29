from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
import re
import string
import torch
import emoji
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tweepy
import os
import time
from dotenv import load_dotenv
from tweepy.errors import TooManyRequests

app = FastAPI()
templates = Jinja2Templates(directory="templates")


MODEL_NAME = "./fine_tuned_bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
sentiment_model = BertForSequenceClassification.from_pretrained(MODEL_NAME)


def map_to_three_classes(label_id):
    if label_id ==0:  # neg
        return "Negative"
    elif label_id == 1:  # nöt
        return "Neutral"
    else:  # poz
        return "Positive"


def clean_text(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace="") 
    text = re.sub(f"[{string.punctuation}]", "", text)  
    return text.strip()


def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return map_to_three_classes(prediction)


vectorizer = TfidfVectorizer(stop_words="english", max_features=50)


load_dotenv()
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)


def fetch_tweets(query, count=10):
    try:
        tweets = client.search_recent_tweets(query=query, max_results=count, tweet_fields=["text"])
        return [tweet.text for tweet in tweets.data] if tweets.data else []
    except TooManyRequests:
        print("Too many requests. Waiting 15 minutes...")
        time.sleep(15 * 60)
        return fetch_tweets(query, count)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def analyze_text(
    request: Request,
    user_text: str = Form(None),
    tweets_query: str = Form(None),  
    file: UploadFile = File(None)
):
    results = []
    texts = []

    if tweets_query:
        tweets = fetch_tweets(tweets_query, count=10)
        for tweet in tweets:
            cleaned_text = clean_text(tweet)
            sentiment_label = analyze_sentiment(cleaned_text)
            texts.append(cleaned_text)
            results.append({
                "text": tweet,
                "cleaned_text": cleaned_text,
                "Sentiment_Label": sentiment_label
            })
            print(f"[TWEET] Original: {tweet} | Cleaned: {cleaned_text} | Sentiment: {sentiment_label}")

    if user_text:
        cleaned_text = clean_text(user_text)
        sentiment_label = analyze_sentiment(cleaned_text)
        texts.append(cleaned_text)
        results.append({
            "text": user_text,
            "cleaned_text": cleaned_text,
            "Sentiment_Label": sentiment_label
        })
        print(f"[TEXT] Original: {user_text} | Cleaned: {cleaned_text} | Sentiment: {sentiment_label}")

    if file:
        contents = await file.read()
        lines = contents.decode("utf-8").split("\n")
        lines = lines[1:] if len(lines) > 1 else lines

        for line in lines:
            if line.strip():
                cleaned_text = clean_text(line)
                sentiment_label = analyze_sentiment(cleaned_text)
                texts.append(cleaned_text)
                results.append({
                    "text": line,
                    "cleaned_text": cleaned_text,
                    "Sentiment_Label": sentiment_label
                })
                print(f"[FILE] Original: {line} | Cleaned: {cleaned_text} | Sentiment: {sentiment_label}")

    # Count sentiment types
    positive_count = sum(1 for r in results if r["Sentiment_Label"] == "Positive")
    negative_count = sum(1 for r in results if r["Sentiment_Label"] == "Negative")
    neutral_count = sum(1 for r in results if r["Sentiment_Label"] == "Neutral")

    print("\n===== Summary of Results =====")
    print(f"Positive: {positive_count}")
    print(f"Negative: {negative_count}")
    print(f"Neutral : {neutral_count}")
    print("================================\n")

    # TF-IDF Analysis (only if more than 1 text)
    if len(texts) > 1:
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        feature_names = vectorizer.get_feature_names_out()

        for i, features in enumerate(tfidf_matrix):
            tfidf_dict = {feature_names[j]: round(features[j], 3) for j in range(len(features)) if features[j] > 0}
            results[i]["TF-IDF_Features"] = tfidf_dict if tfidf_dict else "No significant words found"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "results": results,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count
    })
