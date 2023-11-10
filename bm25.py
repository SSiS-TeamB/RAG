from rank_bm25 import BM25Okapi
from document import BaseDBLoader

## corpus 생성까지
loader = BaseDBLoader()

document = loader.load(is_regex=True, is_split=False)
# print(document)
corpus = loader.get_corpus()

v_list = list(corpus.values())

corpus1 = []
for i in range(len(v_list)):
    corpus1.append(v_list[i])
def tockenizer(sent):
    return sent.split(" ")

tockenized_corpus = [tockenizer(doc) for doc in corpus1]
bm25 = BM25Okapi(tockenized_corpus)

query = str(input())
tokenized_query = tockenizer(query)

doc_scores = bm25.get_scores(tokenized_query)

results= bm25.get_top_n(tokenized_query, corpus1, n=3)

for i, res in enumerate(results):
    print(i+1, res)
    print('$'*20)
    print()