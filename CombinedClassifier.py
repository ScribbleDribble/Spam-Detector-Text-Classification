from nltk.classify import ClassifierI
import statistics


""" we use all the classifiers to classify as 
true or false and the mode classification 
would be chosen """
class CombinedClassifier(ClassifierI):

    poll = []

    classification_confidence = []

    def __init__(self, *classifiers):
        self.classifiers = classifiers

    #overwrite classifier method so we can call all classify with all classifiers in one method
    def classify(self, features):

        for classifier in self.classifiers:

            # the vote will either be T or F
            vote = classifier.classify(features)
            self.poll.append(vote)


        try :
            mode = statistics.mode(self.poll)
            self.classification_confidence.append(self.poll.count(mode)/len(self.poll))
            #reset poll
            self.poll.clear()
            return mode

        except Exception:
            print("Even no. of classifiers entered", Exception)



    #see ratio of algorithms that agree with each other
    def getVoteConfidence(self, features):

        mode = 0

        for classifier in self.classifiers:
            # the vote will either be T or F
            vote = classifier.classify(features)
            self.poll.append(vote)

            mode = statistics.mode(vote)

            # reset poll
            self.poll.clear()

        return self.poll.count(mode) / len(self.poll)





