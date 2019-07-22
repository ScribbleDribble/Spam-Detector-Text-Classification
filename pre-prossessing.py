import csv
from nltk.tokenize import word_tokenize
import nltk
import random

class PreProssessing():

    chat_instances = []
    all_words = ""

    def __init__(self, *files):
        self.files = files

    def readFile(self):

        string = ""

        for file in self.files:

            with open(file, encoding="utf8") as csv_f:
                csvReader = csv.reader(csv_f)

                for line in csvReader:
                    string += line[3] + " "

                print(string)


    def tokenise(self, string):
        self.all_words = word_tokenize(string)
#                self.all_words = word_tokenize(self.all_words)
#                print(self.all_words)


    def findFeatures(self):
#        words_info = nltk.FreqDist(self.all_words)

        top_words = [w[0] for w in words_info.most_common(3000)]


        random.shuffle(top_words)

        print(words_info.most_common())

        words = set(top_words)
        feature_set = {}







pp = PreProssessing("data/Youtube05-Shakira.csv")
pp.readFile()
#pp.findFeatures()
