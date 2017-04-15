#  -*- coding: utf-8 -*-
import sys
import nltk
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.dependencygraph import DependencyGraph
import requests
# import sys
# import codecs
#
# if sys.stdout.encoding != 'cp850':
#   sys.stdout = codecs.getwriter('cp850')(sys.stdout.buffer, 'strict')
# if sys.stderr.encoding != 'cp850':
#   sys.stderr = codecs.getwriter('cp850')(sys.stderr.buffer, 'strict')

# reload(sys)
# sys.setdefaultencoding('utf8')

TEST_DATA_PATH = "../data/test.tsv"
TRAIN_DATA_PATH = "../data/train.tsv"
# path_to_jar = "../stanford-parser-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser.jar"
# path_to_models_jar = "../stanford-parser-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar"
#dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")



def is_reachable(tree, e1, e2):
    E2 = {}
    dep_graph = next(tree)
    reachable = 0
    start = E2
    for index in range(len(dep_graph.nodes)):
        node = dep_graph.nodes[index]
        if(node['word'] == e2):
            start = node
            break

    while start['word'] is not None:
        if(start['word'] == e1):
            reachable = 1
            break
        else:
            start = dep_graph.nodes[start['head']]
    print(reachable)
    return reachable





def get_dep_feat(intermediate_text, person, institution):
    features = []

    # making the person entity as the last name of the person for simplicity
    person = person.split()
    person = person[len(person) - 1]

    # making the institution the first word of the institution name for simplicity
    institution = institution.split()
    institution = institution[len(institution) - 1]

    sentence = person + " " + intermediate_text + " " + institution

    # dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    result = dep_parser.raw_parse(sentence)

    # Feature 1 : is e1 is reachable from e2 if we traverse in the dependency tree
    features.extend([is_reachable(result, person, institution)])

    return features


def parse_data(train_data, test_data):
    """
    Input: path to the data file
    Output: (1) a list of tuples, one for each instance of the data, and
            (2) a list of all unique tokens in the data

    Parses the data file to extract all instances of the data as tuples of the form:
    (person, institution, judgment, full snippet, intermediate text)
    where the intermediate text is all tokens that occur between the first occurrence of
    the person and the first occurrence of the institution.

    Also extracts a list of all tokens that appear in the intermediate text for the
    purpose of creating feature vectors.
    """

    all_tokens = []
    data = []
    #for fp in [train_data, test_data]:
    for fp in [test_data]:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                institution, person, snippet, intermediate_text, judgment = line.split("\t")
                judgment = judgment.strip()
                intermediate_text = intermediate_text.lower()
                intermediate_text = intermediate_text.replace("/", " ")
                intermediate_text = intermediate_text.replace("-", " ")
                intermediate_text = intermediate_text.replace("(", " ")
                intermediate_text = intermediate_text.replace(")", " ")
                intermediate_text = intermediate_text.replace(".", " ")
                intermediate_text = intermediate_text.replace("?", " ")
                intermediate_text = intermediate_text.replace(",", " ")
                intermediate_text = intermediate_text.replace(";", " ")
                intermediate_text.strip()
                # Build up a list of unique tokens that occur in the intermediate text
                # This is needed to create BOW feature vectors
                # tokens = intermediate_text.split()
                tokens = nltk.word_tokenize(intermediate_text)
                for t in tokens:
                    t = t.lower()
                    if t not in all_tokens:
                        all_tokens.append(t)
                data.append((person, institution, judgment, snippet, intermediate_text))
    return data, all_tokens


def create_feature_vectors(data, all_tokens):
    """
    Input: (1) The parsed data from parse_data()
             (2) a list of all unique tokens found in the intermediate text
    Output: A list of lists representing the feature vectors for each data instance

    Creates feature vectors from the parsed data file. These features include
    bag of words features representing the number of occurrences of each
    token in the intermediate text (text that comes between the first occurrence
    of the person and the first occurrence of the institution).
    This is also where any additional user-defined features can be added.
    """
    feature_vectors = []
    for instance in data:
        # BOW features
        # Gets the number of occurrences of each token
        # in the intermediate text
        feature_vector = [0 for t in all_tokens]
        intermediate_text = instance[4]
        # tokens = intermediate_text.split()
        tokens = nltk.word_tokenize(intermediate_text)
        for token in tokens:
            index = all_tokens.index(token.lower())
            feature_vector[index] += 1

        ### ADD ADDITIONAL FEATURES HERE ###
        feature_vector.extend(get_dep_feat(intermediate_text, instance[0].lower(), instance[1].lower()))

        # Class label
        judgment = instance[2]
        feature_vector.append(judgment)

        feature_vectors.append(feature_vector)
    return feature_vectors


def generate_arff_file(feature_vectors, all_tokens, out_path):
    """
    Input: (1) A list of all feature vectors for the data
             (2) A list of all unique tokens that occurred in the intermediate text
             (3) The name and path of the ARFF file to be output
    Output: an ARFF file output to the location specified in out_path

    Converts a list of feature vectors to an ARFF file for use with Weka.
    """
    with open(out_path, 'w') as f:
        # Header info
        f.write("@RELATION institutions\n")
        for i in range(len(all_tokens)):
            f.write("@ATTRIBUTE token_{} INTEGER\n".format(i))

        ### SPECIFY ADDITIONAL FEATURES HERE ###
        # For example: f.write("@ATTRIBUTE custom_1 REAL\n")

        # Classes
        f.write("@ATTRIBUTE class {yes,no}\n")

        # Data instances
        f.write("\n@DATA\n")
        for fv in feature_vectors:
            features = []
            for i in range(len(fv)):
                value = fv[i]
                if value != 0:
                    features.append("{} {}".format(i, value))
            entry = ",".join(features)
            f.write("{" + entry + "}\n")


def save_the_dep_graph_as_conll():
    data, all_tokens = parse_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    count = 0
    with open("test.conll",'w', encoding='utf-8') as fout:
        for instance in data:
            count += 1
            intermediate_text = instance[4]
            person = instance[0]
            institution = instance[1]

            # making the person the first word of the person name for simplicity
            person = person.split()
            person = person[0]

            # making the institution the first word of the institution name for simplicity
            institution = institution.split()
            institution = institution[0]

            sentence = person + " " + intermediate_text + " " + institution
            result = requests.post('http://localhost:9000/?properties={%22annotators%22%3A%22depparse%2Cssplit%2Cpos%22%2C%22outputFormat%22%3A%22conll%22}', data=sentence.encode('utf-8'))
            #print(result.text)

            #result = next(dep_parser.raw_parse(sentence))
            print(count)
            fout.write(result.text + "\n")

save_the_dep_graph_as_conll()

def load_dep_trees(finput):
    out = DependencyGraph.load('train.conll')
    print("loaded")

#load_dep_trees("train.conll")
# if __name__ == "__main__":
    # data, all_tokens = parse_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    # feature_vectors = create_feature_vectors(data, all_tokens)
    # generate_arff_file(feature_vectors[:6000], all_tokens, "../data/token_arff_clean/train.arff")
    # generate_arff_file(feature_vectors[6000:], all_tokens, "../data/token_arff_clean/test.arff")