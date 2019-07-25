import pickle
# work around
import pre_prossessing
import statistics
import nltk
from nltk.classify import ClassifierI

from CombinedClassifier import CombinedClassifier as Comb

class CombinedClassifier(ClassifierI):

    poll = []
    classification_confidence = []

    def __init__(self, *classifiers):
        self.classifiers = classifiers

    def classify(self, features):
        for classifier in self.classifiers:
            # the vote will either be T or F
            vote = classifier.classify(features)
            self.poll.append(vote)

        try:
            mode = statistics.mode(self.poll)
            self.classification_confidence.append(self.poll.count(mode) / len(self.poll))
            # reset poll
            self.poll.clear()
            return mode

        except Exception:
            print("Even no. of classifiers entered", Exception)


def getFeatures(user_comment):

    f = open("Classifiers/top_words.pickle", "rb")
    top_words = pickle.load(f)

    featuresDict = {}

    for w in top_words:
        featuresDict[w] = w in user_comment

    return featuresDict

def result(text):
    feats = getFeatures(text)
    return Comb.classify(feats)

if __name__ == '__main__':

    f = open("Classifiers/LinearSVC.pickle", "rb")
    linearSVC = pickle.load(f)
    f.close()

    f = open("Classifiers/BernoulliNB.pickle", "rb")
    bernoulliNB = pickle.load(f)
    f.close()

    f = open("Classifiers/LinearSVC.pickle", "rb")
    linearSVC = pickle.load(f)
    f.close()

    f = open("Classifiers/LogisticRegression.pickle", "rb")
    logistic_regression = pickle.load(f)
    f.close()

    f = open("Classifiers/MultinomialMB.pickle", "rb")
    multinomialNB = pickle.load(f)
    f.close()

    f = open("Classifiers/SVC.pickle", "rb")
    svc = pickle.load(f)
    f.close()

    comb = CombinedClassifier(bernoulliNB, linearSVC, logistic_regression, logistic_regression, multinomialNB, svc)




    #print(result("hi can you please subscribe to my channel"))
    #print(result("The 4 people who like the clone-wars Visible anger."))
    #print(result("The narrator voice is spot on! Haha I guess I'm one of the five people who liked clone wars."))
    #print(result("C'mon Dice just make Battlefield Star Wars, yeah? Plz listen to Jack, Dice"))
    print(result("Watch Maroon 5's latest 2nd single from V (It Was Always You) www.youtube. com/watch?v=TQ046FuAu00"))
