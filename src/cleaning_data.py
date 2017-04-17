from nltk.parse.stanford import StanfordDependencyParser

dep_parser=StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

result = dep_parser.raw_parse('I shot an elephant in my sleep')
for edges in result:
    print(edges)