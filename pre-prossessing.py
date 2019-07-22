import csv
from nltk.tokenize import word_tokenize
import nltk
import nltk.classify
from nltk.corpus import stopwords
import math

import random
import re



class PreProssessing():

    chat_instances = []
    all_words = ""


    def __init__(self, *files):
        self.files = files

    def readFile(self):

        string = ""

        for file in self.files:

            with open(file, encoding="utf-8-sig") as csv_f:
                csvReader = csv.reader(csv_f)

                for line in csvReader:
                    # i will change this once i figure out a better way i.e StringBuilder in Java
                    string += line[3].lower()
                    # fill list with a tuple containing the comment alongside its label

                    self.chat_instances.append((line[3].lower(), line[4]))





                self.all_words = string



    def tokenise(self):
        s = re.findall(r"""[a-z][a-z]+""", self.all_words)

        self.all_words = s


        #print(self.all_words)

    def getFeatures(self, user_comment):
        words_info = nltk.FreqDist(self.all_words)

        top_words = [w[0] for w in words_info.most_common(1000)]

        words = set(user_comment)

        featuresDict = {}

        for w in top_words:
            featuresDict[w] = w in user_comment

        return featuresDict

    def getFeatureSet(self):
        random.shuffle(self.chat_instances)
        feature_set = [(self.getFeatures(comment), label) for (comment, label) in self.chat_instances]

        return feature_set

pp = PreProssessing("data/Youtube04-Eminem.csv", "data/Youtube03-LMFAO.csv", "data/Youtube02-KatyPerry.csv")
pp.readFile()
pp.tokenise()
feature_set = pp.getFeatureSet()

print(feature_set)

training_data = feature_set[:  math.floor(len(feature_set) * 0.8)]
testing_data = feature_set[math.floor(len(feature_set) * 0.8):]

classifier = nltk.NaiveBayesClassifier.train(training_data)
print("algo acc: ", nltk.classify.accuracy(classifier, testing_data))
classifier.show_most_informative_features(15)