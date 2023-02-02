import twint, pickle, re, nltk
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

tokenizer = pickle.load(open("C:/Users/NyteStar/PycharmProjects/data/tokenizer.pkl", "rb"))
model = keras.models.load_model("C:/Users/NyteStar/PycharmProjects/data/model.h5")

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
SEQUENCE_LENGTH = 300

nltk.download('stopwords')
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


def preprocess(text, stem):
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)

    return " ".join(tokens)


def decode_sentiment(score, include_neutral):
    label = "NEUTRAL"
    if score <= SENTIMENT_THRESHOLDS[0]:
        label = "NEGATIVE"
    elif score >= SENTIMENT_THRESHOLDS[1]:
        label = "POSITIVE"

    return label


def predict(text):
    x = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=SEQUENCE_LENGTH
    )

    return float(model.predict([x])[0])


def twint_search(keyword, limit):
    tweets = []

    c = twint.Config()
    c.Search = keyword
    c.Limit = limit
    c.Lang = "en"
    # c.Custom["tweet"] = ["username", "date", "tweet"]
    c.Hide_output = True
    c.Store_object = True
    c.Store_object_tweets_list = tweets

    twint.run.Search(c)

    return tweets


def predictOnInput(keyword, stem, include_neutral, twintlimit=40):
    if int(twintlimit) > 301:
        twintlimit = 300
    tweets = twint_search(keyword, twintlimit)

    t = []
    negatives = []
    positives = []
    neutral = []
    for tweet in tweets:
        processed_text = preprocess(tweet.tweet, stem)
        score = predict(processed_text)
        label = decode_sentiment(score, include_neutral)
        if label == "NEGATIVE":
            negatives.append({
                'username': tweet.username,
                'tweet': tweet.tweet,
                'link': tweet.link,
                'score': score,
                'label': decode_sentiment(score, include_neutral),
                'date': tweet.datestamp,
                'processed_text': processed_text
            })
        elif label == "POSITIVE":
            positives.append({
                'username': tweet.username,
                'tweet': tweet.tweet,
                'link': tweet.link,
                'score': score,
                'label': decode_sentiment(score, include_neutral),
                'date': tweet.datestamp,
                'processed_text': processed_text
            })
        else:
            neutral.append({
                'username': tweet.username,
                'tweet': tweet.tweet,
                'link': tweet.link,
                'score': score,
                'label': decode_sentiment(score, include_neutral),
                'date': tweet.datestamp,
                'processed_text': processed_text
            })

    return negatives, positives, neutral
