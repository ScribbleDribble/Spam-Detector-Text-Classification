import csv
import nltk
import nltk.classify
import math

import random
import re

import pickle

from combinedclassifier import CombinedClassifier

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier #stochastic gradient descent
from sklearn.svm import SVC, LinearSVC, NuSVC

import pickle

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

    def getFeatures(self, user_comment):
        words_info = nltk.FreqDist(self.all_words)

        top_words = [w[0] for w in words_info.most_common(1800)]

        save_all_words = open("Classifiers/top_words.pickle", "wb")
        pickle.dump(top_words, save_all_words)
        save_all_words.close()

        featuresDict = {}

        for w in top_words:
            featuresDict[w] = w in user_comment

        return featuresDict

    def getFeatureSet(self):
        random.shuffle(self.chat_instances)
        feature_set = [(self.getFeatures(comment), label) for (comment, label) in self.chat_instances]

        return feature_set

pp = PreProssessing("data/Youtube04-Eminem.csv",
                    "data/Youtube03-LMFAO.csv",
                    "data/Youtube02-KatyPerry.csv",
                    "data/Youtube05-Shakira.csv")
pp.readFile()
pp.tokenise()
feature_set = pp.getFeatureSet()

if __name__ == '__main__':
    training_data = feature_set[:  math.floor(len(feature_set) * 0.8)]
    testing_data = feature_set[math.floor(len(feature_set) * 0.8):]

    classifier = nltk.NaiveBayesClassifier.train(training_data)
    #save_classifier = open("Classifiers/NaiveBayes.pickle", "wb")
    #pickle.dump(classifier, save_classifier)
    #save_classifier.close()

    multinomialMB = SklearnClassifier(MultinomialNB())
    multinomialMB.train(training_data)
    #save_classifier = open("Classifiers/MultinomialMB.pickle", "wb")
    #pickle.dump(multinomialMB, save_classifier)
    #save_classifier.close()

    logistic_regression = SklearnClassifier(LogisticRegression())
    logistic_regression.train(training_data)
    #save_classifier = open("Classifiers/LogisticRegression.pickle", "wb")
    #pickle.dump(logistic_regression, save_classifier)
    #save_classifier.close()

    bernoulliNB = SklearnClassifier(BernoulliNB())
    bernoulliNB.train(training_data)
    #save_classifier = open("Classifiers/BernoulliNB.pickle", "wb")
    #pickle.dump(bernoulliNB, save_classifier)
    #save_classifier.close()

    svm = SklearnClassifier(SVC())
    svm.train(training_data)
    #save_classifier = open("Classifiers/SVC.pickle", "wb")
    #pickle.dump(svm, save_classifier)
    #save_classifier.close()

    linear_SVC = SklearnClassifier(LinearSVC())
    linear_SVC.train(training_data)
    #save_classifier = open("Classifiers/LinearSVC.pickle", "wb")
    #pickle.dump(linear_SVC, save_classifier)
    #save_classifier.close()

    sgdc = SklearnClassifier(SGDClassifier())
    sgdc.train(training_data)
    #save_classifier = open("Classifiers/sgdc.pickle", "wb")
    #pickle.dump(sgdc, save_classifier)
    #save_classifier.close()

    comb = CombinedClassifier(classifier, multinomialMB, logistic_regression, svm, bernoulliNB, linear_SVC, sgdc)

    print("algo acc: ", nltk.classify.accuracy(comb, testing_data))
    classifier.show_most_informative_features(50)


