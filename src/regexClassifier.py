import nltk
from nltk import RegexpParser
from nltk import pos_tag
"""

patterns found
[graduation/graduated from/graduated from the, education/educated at,
attended/attended the, graduate of the, matriculated at, enrolled at the/enrolled at,
 attended]
"""




def predicting_relations(finput):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with open(finput, 'r') as input:
        for line in input:
            splitted_txt = line.split("\t")
            intermediate_txt = splitted_txt[3]
            train_label = splitted_txt[4]
            if isMatch(intermediate_txt):
                if train_label == "yes":
                    TP += 1
                else:
                    FP += 1
            else:
                if train_label == "No":
                    TN += 1
                else:
                    FN += 1
    print("True Positive : " + str(TP))
    print("False Positive : " + str(FP))
    print("True Negetive : " + str(TN))
    print("False Negative : " + str(FP))

def isMatch(text):
    pattern = """
    P : {.*[graduated from].*}
    """
    cp = nltk.RegexpParser(pattern)
    result = cp.parse(pos_tag(text))
    print(result)

#predicting_relations("../data/testFile")