import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
import re
import nltk
nltk.download("stopwords")  # Corpus klasörüne indiriliyor
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv(r"gender_classification.csv", encoding="latin1")

df = pd.concat([df.gender, df.description], axis=1)

df.dropna(inplace=True, axis=0)

# Male = 0 || Female = 1
df.gender = [1 if each == "female" else 0 for each in df.gender]

# Cleaning Data *Regular Expression
description_list = []
for description in df.description:
    description = re.sub("[^a-zA-Z]", " ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    # description = [word for word in description if not word in set(
    #     stopwords.words("english"))]
    lemma = nltk.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]

    description = " ".join(description)
    description_list.append(description)

# Bef of words

max_features = 5000

count_vec = CountVectorizer(max_features=max_features, stop_words="english")
sparce_matrix = count_vec.fit_transform(description_list)

# print("En çok kullanılan {} kelimeler: {}".format(
#     max_features, count_vec.get_feature_names()))

y = df.iloc[:, 0].values  # male or female classes
x = sparce_matrix

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=25)

# Naive bayes

NB = GaussianNB()
NB.fit(x_train.toarray(), y_train)

y_pred = NB.predict(x_test.toarray())
print("The model accuracy is: ", NB.score(y_pred.reshape(-1, 1), y_test))


""" 
! Simple tests
# %% Simpel RE Example
simple_description = df.description[4]
true1_description = re.sub("[^a-zA-Z]", " ", simple_description)
true1_description = true1_description.lower()

# %% stopwords (irrelavent words)
# natural language tool kit (stop words)

# true1_description=true1_description.split() # veya tokenizer kullanabiliriz ve bu shouldn't u 2 ye ayırır ama split ayırmaz
true1_description = nltk.word_tokenize(true1_description)

relevant_description = [word for word in true1_description if not word in set(
    stopwords.word("english"))]

# %%emmatazation Loved =>> Love
lemma = nltk.WordNetLemmatizer()

relevant_description = [lemma.lemmatize(word) for word in relevant_description]


# descp = " ".join(relevant_description)

"""
