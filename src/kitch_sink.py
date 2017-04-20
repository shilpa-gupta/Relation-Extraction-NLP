
import sys
import nltk
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.dependencygraph import DependencyGraph

TEST_DATA_PATH = "../data/test.tsv"
TRAIN_DATA_PATH = "../data/train.tsv"

def parse_data(train_data, test_data):
    all_tokens = []
    data = []
    for fp in [train_data, test_data]:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                institution, person, snippet, intermediate_text, judgment = line.split("\t")
                judgment = judgment.strip()
                intermediate_text.strip()
                # Build up a list of unique tokens that occur in the intermediate text
                # This is needed to create BOW feature vectors
                tokens = nltk.word_tokenize(intermediate_text)
                for t in tokens:
                    t = t.lower()
                    if t not in all_tokens:
                        all_tokens.append(t)
                data.append((person, institution, judgment, snippet, intermediate_text))
    return data, all_tokens

def load_dep_feat():
    all_feats = []
    with open("../data/kitchen_sink.csv", 'r', encoding="utf-8") as input:
        for line in input:
            tokens = line.split(",")
            tokens[len(tokens)-1] = tokens[len(tokens)-1].strip()
            tokens = [int(i) for i in tokens]
            all_feats.append(tokens)
    return all_feats

# load_dep_feat()

def create_feature_vectors(data, all_tokens):
    feature_vectors = []
    i = 0
    dep_feat = load_dep_feat()
    for idx,instance in enumerate(data):
        feature_vector = []

        ### ADD ADDITIONAL FEATURES HERE ###
        feature_vector.extend(dep_feat[idx])

        # Class label
        judgment = instance[2]
        feature_vector.append(judgment)

        feature_vectors.append(feature_vector)
        i += 1
    return feature_vectors


def generate_arff_file(feature_vectors, all_tokens, out_path):
    with open(out_path, 'w') as f:
        # Header info
        f.write("@RELATION institutions\n")
        for i in range(506):
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

if __name__ == "__main__":
    data, all_tokens = parse_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    feature_vectors = create_feature_vectors(data, all_tokens)
    generate_arff_file(feature_vectors[:6000], all_tokens, "../data/kitch_sink_arff/train.arff")
    generate_arff_file(feature_vectors[6000:], all_tokens, "../data/kitch_sink_arff/test.arff")
