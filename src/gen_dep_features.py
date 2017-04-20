"""
this function load the conll trees which has been generated for train and test data using the stanford corenlp dependency parser
"""
def load_dep_trees(finput):
    tree_list = []
    with open(finput, 'r', encoding='utf-8') as input:
        for line in input.read().split("\n\n"):
            tree = []
            edges = line.split("\n")
            for edge in edges:
                node = {}
                tokens = edge.split("\t")
                node['index'] = tokens[0]
                node['word'] = tokens[1]
                node['tag'] = tokens[3]
                node['head'] = tokens[6]
                node['label'] = tokens[7]
                node['visited'] = False
                tree.append(node)
            tree_list.append(tree)
    return tree_list

# load_dep_trees("../data/total_dep_trees.conll")

"""
this function mark all the nodes of the input tree, unvisited
"""
def unvisit_tree(tree):
    for node in tree:
        node['visited'] = False

"""
this function gives the length of the left path and right path that is length of the path from E1 to root and E2 to root
"""
def gen_length_left_path():
    tree_list = load_dep_trees("../data/total_dep_trees.conll")
    with open("../data/left_right_path_len.csv", 'w', encoding="utf-8") as output:
        for tree in tree_list:
            unvisit_tree(tree)
            person_node = tree[0]
            inst_node = tree[len(tree)-1]

            left_path_len = 0
            start = person_node
            while start['head'] != '0' and start['visited'] == False:
                start['visited'] = True
                left_path_len += 1
                start = tree[int(start['head'])-1]

            right_path_len = 0
            unvisit_tree(tree)
            start = inst_node
            while start['head'] != '0' and start['visited'] == False:
                start['visited'] = True
                right_path_len += 1
                start = tree[int(start['head'])-1]
            print("left path len : ")
            print(left_path_len)
            print("right path len : ")
            print(right_path_len)
            output.write(",".join([str(x) for x in [left_path_len, right_path_len]]) + "\n")

#gen_length_left_path()

"""
this function gives the list of all tags present, to make the BOW representation of the tags
"""
def extract_all_tags(tree_list):
    tags = []
    for tree in tree_list:
        for node in tree:
            if node['tag'] not in tags:
                tags.append(node['tag'])

    return tags

"""
this function gives the list of all labels present in the given data for the purpose of BOW representation of the 
labels
"""
def extract_all_labels(tree_list):
    labels = []
    for tree in tree_list:
        for node in tree:
            if node['label'] not in labels:
                labels.append(node['label'])
    return labels

"""
returns the bag of word representation of the tags and labels
"""
def bow_of_paths():
    tree_list = load_dep_trees("../data/total_dep_trees.conll")
    all_tags = extract_all_tags(tree_list)
    cnt = 0
    with open("../data/left_path_bow_tags.csv",'w',encoding='utf-8') as input_left:
        with open("../data/right_path_bow_tags.csv", 'w', encoding='utf-8') as input_right:
            for tree in tree_list:
                print(cnt)
                cnt += 1
                left_path_bow = [0 for t in all_tags]
                right_path_bow = [0 for t in all_tags]
                person_node = tree[0]
                inst_node = tree[len(tree) - 1]
                unvisit_tree(tree)
                start = person_node
                while start['head'] != '0' and start['visited'] == False:
                    start['visited'] = True
                    index = all_tags.index(start['tag'])
                    left_path_bow[index] += 1

                unvisit_tree(tree)
                start = inst_node
                while start['head'] != '0' and start['visited'] == False:
                    start['visited'] = True
                    index = all_tags.index(start['tag'])
                    right_path_bow[index] += 1
                input_left.write(",".join(str(x) for x in left_path_bow) + "\n")
                input_right.write(",".join(str(x) for x in right_path_bow) + "\n")


# bow_of_paths()

"""
this function extract if the entity1 and entity2 are connected or not
it takes 1 if they both have same root 0 otherwise
"""
def extract_if_entities_are_connected():
    tree_list = load_dep_trees("../data/total_dep_trees.conll")
    with open("../data/isConnected.csv", 'w', encoding="utf-8") as output:
        for tree in tree_list:
            person_node = tree[0]
            inst_node = tree[len(tree) - 1]

            root1 = person_node
            unvisit_tree(tree)
            while root1['head'] != '0' and root1['visited'] == False:
                root1['visited'] = True
                root1 = tree[int(root1['head'])-1]

            root2 = inst_node
            unvisit_tree(tree)
            while root2['head'] != '0' and root2['visited'] == False:
                root2['visited'] = True
                root2 = tree[int(root2['head']) - 1]

            if root1['index'] == root2['index']:
                output.write(str(1) + "\n")
            else:
                output.write(str(0) + "\n")

#extract_if_entities_are_connected()