import nltk

TEST_DATA_PATH = "../data/test.tsv"
TRAIN_DATA_PATH = "../data/train.tsv"
ID_LEN = 11


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
    for fp in [train_data, test_data]:
        with open(fp, 'r', encoding='utf8') as f:
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
                #tokens = intermediate_text.split()
                tokens = nltk.word_tokenize(intermediate_text)
                for t in tokens:
                    t = t.lower()
                    if t not in all_tokens:
                        all_tokens.append(t)
                data.append((person, institution, judgment, snippet, intermediate_text))
    return data, all_tokens


def load_clusters(fin):
    with open(fin, "r", encoding="utf8") as finput:
        dict = {}
        clusters = []
        for line in finput:
            token = line.split()
            clust_id = token[0][0:ID_LEN]
            print(clust_id)
            word = token[1]
            dict[word] = clust_id
            if clust_id not in clusters:
                clusters.append(clust_id)
    return dict, clusters
            


cluster_dict = load_clusters("../data/paths")

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

    cluster_dict, clusters = load_clusters("../data/paths")
    feature_vectors = []
    for instance in data:
        # BOW features
        # Gets the number of occurrences of each token
        # in the intermediate text

        feature_vector = [0 for t in clusters]
        intermediate_text = instance[4]
        #tokens = intermediate_text.split()
        tokens = nltk.word_tokenize(intermediate_text.lower())
        for token in tokens:
            if token in cluster_dict.keys():
                index = clusters.index(cluster_dict[token.lower()])
                feature_vector[index] += 1

        ### ADD ADDITIONAL FEATURES HERE ###

        # Class label
        judgment = instance[2]
        feature_vector.append(judgment)

        feature_vectors.append(feature_vector)
    return feature_vectors, clusters


def generate_arff_file(feature_vectors, clusters, out_path):
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
        for i in range(len(clusters)):
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
    feature_vectors, clusters = create_feature_vectors(data, all_tokens)
    generate_arff_file(feature_vectors[:6000], clusters, "../data/cluster_arff/id_len_11/train.arff")
    generate_arff_file(feature_vectors[6000:], clusters, "../data/cluster_arff/id_len_11/test.arff")