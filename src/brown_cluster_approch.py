import nltk

TEST_DATA_PATH = "../data/test.tsv"
TRAIN_DATA_PATH = "../data/train.tsv"
ID_LEN = 11


def parse_data(train_data, test_data):
    all_tokens = []
    data = []
    for fp in [train_data, test_data]:
        with open(fp, 'r', encoding='utf8') as f:
            for line in f:
                institution, person, snippet, intermediate_text, judgment = line.split("\t")
                judgment = judgment.strip()
                intermediate_text.strip()
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
        
# cluster_dict = load_clusters("../data/paths")

def extracting_intermediate_text(finput1, finput2, foutput):
    with open(foutput, 'w', encoding="utf8") as output:
        with open(finput1, 'r', encoding="utf8") as input:
            for line in input:
                line = line.lower()
                text = line.split("\t")[2]
                text = text.lower()
                tokens = nltk.word_tokenize(text)
                output.write(" ".join(tokens) + '\n')
        with open(finput2, 'r', encoding="utf8") as input:
            for line in input:
                line = line.lower()
                text = line.split("\t")[3]
                text = text.lower()
                tokens = nltk.word_tokenize(text)
                output.write(" ".join(tokens) + '\n')
# extracting_intermediate_text("../data/train.tsv", "../data/test.tsv", "../data/text.txt")

def create_feature_vectors(data, all_tokens):
    cluster_dict, clusters = load_clusters("../data/paths")
    feature_vectors = []
    for instance in data:
        feature_vector = [0 for t in clusters]
        intermediate_text = instance[4]
        tokens = nltk.word_tokenize(intermediate_text.lower())
        for token in tokens:
            if token in cluster_dict.keys():
                index = clusters.index(cluster_dict[token.lower()])
                feature_vector[index] += 1
        judgment = instance[2]
        feature_vector.append(judgment)

        feature_vectors.append(feature_vector)
    return feature_vectors, clusters


def generate_arff_file(feature_vectors, clusters, out_path):
    with open(out_path, 'w') as f:
        # Header info
        f.write("@RELATION institutions\n")
        for i in range(len(clusters)):
            f.write("@ATTRIBUTE token_{} INTEGER\n".format(i))
        f.write("@ATTRIBUTE class {yes,no}\n")
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
    generate_arff_file(feature_vectors[:6000], clusters, "../data/cluster_arff/id_len_3/train.arff")
    generate_arff_file(feature_vectors[6000:], clusters, "../data/cluster_arff/id_len_3/test.arff")