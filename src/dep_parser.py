import json, jsonrpclib
from pprint import pprint
#server = jsonrpclib.Server("http://localhost:8080")
#print(json.loads(server.parse('John loves Mary.'))) 


from nltk.parse.stanford import StanfordDependencyParser
dep_parser=StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
print [parse.tree() for parse in dep_parser.raw_parse("The quick brown fox jumps over the lazy dog.")]