import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('wordnet')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from train import TrainData
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


df = pd.read_excel("Data.xlsx", sheet_name="Sheet1")
df = df.drop(['Label'], axis=1)
df = df.drop(['Date of review'], axis=1)
df.dropna(axis=0, inplace=True)
df.isna().sum()
df.isnull().sum(axis=1)
df.loc[df.SENTIMENT == "positive", "SENTIMENT"] = 1
df.loc[df.SENTIMENT == "Negative", "SENTIMENT"] = 0
df.SENTIMENT.unique()

feedback, sentiment = list(df['Actual Feedback']), list(df['SENTIMENT'])

vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
t = TrainData()
processedtext = t.preprocess(feedback)

sns.countplot('SENTIMENT', data=df)

t.train(p=processedtext, s=sentiment)

model = t.bnb()
# model = t.svc()
# model = t.lr()