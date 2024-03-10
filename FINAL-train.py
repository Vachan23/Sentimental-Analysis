import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('wordnet')
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


class TrainData:
    def train(self,p,s):
        global X_train, X_test, y_train, y_test, vectoriser
        X_train, X_test, y_train, y_test = train_test_split(p, s, test_size=0.05, random_state=0)
        vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        vectoriser.fit(X_train)
        X_train = vectoriser.transform(X_train)
        X_test = vectoriser.transform(X_test)

    def model_Evaluate(self, model):
        y_pred = model.predict(X_test)
        score = int(accuracy_score(y_test, y_pred) * 100)
        round(score, 2)
        print(f"Accuracy Score of {model} : {score}%")

    def bnb(self):
        BNBmodel = BernoulliNB(alpha=2)
        BNBmodel.fit(X_train, y_train)
        return self.model_Evaluate(BNBmodel)

    def svc(self):
        SVCmodel = LinearSVC()
        SVCmodel.fit(X_train, y_train)
        return self.model_Evaluate(SVCmodel)

    def lr(self):
        LRmodel = LogisticRegression(C=2, max_iter=1000)
        LRmodel.fit(X_train, y_train)
        return self.model_Evaluate(LRmodel)

    def preprocess(self, textdata):
        processedText = []

        wordLemm = WordNetLemmatizer()
        sent = "[^a-zA-Z0-9]"

        for text in textdata:
            text = text.lower()
            text = re.sub(sent, " ", text)
            words = ''
            for word in text.split():
                if len(word) > 1:
                    word = wordLemm.lemmatize(word)
                    words += (word + ' ')
            processedText.append(words)
        return processedText


