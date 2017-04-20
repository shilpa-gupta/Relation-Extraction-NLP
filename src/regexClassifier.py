import re

def isMatch(text):
    regex = [r'^.* graduated .*', r'^.* studied .*', r'^.* attended .*', r'^.* received .*', r'^.* educated .*', r'^.* degree .*']
    for item in regex:
        if re.match(item, text.lower()) is not None:
            return True
    return False

"""
matching the regular expressions and calculating the true posititve, true negative, 
false positive and false negative values for the test data
"""
def predicting_relations(finput):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    tp_out = open("../data/TP_Samples.txt", 'w', encoding="utf-8")
    fp_out = open("../data/FP_Samples.txt", 'w', encoding="utf-8")
    tn_out = open("../data/TN_Samples.txt", 'w', encoding="utf-8")
    fn_out = open("../data/FN_Samples.txt", 'w', encoding="utf-8")
    with open(finput, 'r', encoding="utf-8") as input:
        for line in input:
            splitted_txt = line.split("\t")
            intermediate_txt = splitted_txt[3]
            train_label = splitted_txt[4]
            train_label = train_label.strip()
            if isMatch(intermediate_txt):
                if train_label == "yes":
                    tp_out.write(intermediate_txt + "\n")
                    TP += 1
                else:
                    fp_out.write(intermediate_txt + "\n")
                    FP += 1
            else:
                if train_label == "no":
                    tn_out.write(intermediate_txt + "\n")
                    TN += 1
                else:
                    fn_out.write(intermediate_txt + "\n")
                    FN += 1
    print("True Positive : " + str(TP))
    print("False Positive : " + str(FP))
    print("True Negetive : " + str(TN))
    print("False Negative : " + str(FP))
    tp_out.close()
    fp_out.close()
    tn_out.close()
    fn_out.close()

predicting_relations("../data/test.tsv")

def analyze_patterns(infile):
    graduated = 0
    studied = 0
    attended = 0
    received = 0
    educated = 0
    degree = 0
    with open(infile, 'r', encoding='utf-8') as input:
        for line in input:
            row = line.split("\t")
            text = row[3]

            if "graduated" in text.lower():
                graduated += 1
            if "studied" in text.lower():
                studied += 1
            if "attended" in text.lower():
                attended += 1
            if "received" in text.lower():
                received += 1
            if "educated" in text.lower():
                educated += 1
            if "degree" in text.lower():
                degree += 1
    print("Graduated : ")
    print(graduated)
    print("studied : ")
    print(studied)
    print("attended : ")
    print(attended)
    print("received : ")
    print(received)
    print("educated : ")
    print(educated)
    print("degree")
    print(degree)

#analyze_patterns("../data/train.tsv")


