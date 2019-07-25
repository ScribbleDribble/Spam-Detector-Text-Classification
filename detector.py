import pickle

from combinedclassifier import CombinedClassifier as Comb

"""Obtain features from input"""
class PrepareData():

    def getFeatures(self,user_comment):

        f = open("Classifiers/top_words.pickle", "rb")
        top_words = pickle.load(f)

        featuresDict = {}

        for w in top_words:
            featuresDict[w] = w in user_comment

        return featuresDict

    def result(self, classifiers, text):
        feats = self.getFeatures(text)
        return classifiers.classify(feats)

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

    prep = PrepareData()
    combined_classifier = Comb(bernoulliNB, linearSVC, logistic_regression, logistic_regression, multinomialNB, svc)

    # examples
    print(prep.result(combined_classifier,"Watch Maroon 5's latest 2nd single from V (It Was Always You) www.youtube. com/watch?v=TQ046FuAu00"))
    print(prep.result(combined_classifier,"that was so damn good"))
    print(prep.result(combined_classifier,"hi can you please subscribe to my channel"))
    # works with all types of general comments
    print(prep.result(combined_classifier,"The 4 people who like the clone-wars Visible anger."))
    print(prep.result(combined_classifier,"The narrator voice is spot on! Haha I guess I'm one of the five people who liked clone wars."))
    print(prep.result(combined_classifier,"C'mon Dice just make Battlefield Star Wars, yeah? Plz listen to Jack, Dice"))
    print(prep.result(combined_classifier,"Watch Maroon 5's latest 2nd single from V (It Was Always You) www.youtube. com/watch?v=TQ046FuAu00"))