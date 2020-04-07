vec_size = 100
window = 10
min_count = 1000
input_file = "output/clean-txt.txt"
output_file = "output/d2v.txt"

import numpy as np
import smart_open
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="utf8") as f:
        for i, line in enumerate(f):
            tokens = simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield TaggedDocument(tokens, [i])

tagged_data = list(read_corpus(input_file))

max_epochs = 100
alpha = 0.025

model_dbow = Doc2Vec(vector_size=vec_size,
                window=window,
                min_count=min_count,
                alpha=alpha,
                min_alpha=alpha,
                dm=0)
model_dm = Doc2Vec(vector_size=vec_size,
                window=window,
                min_count=min_count,
                alpha=alpha,
                min_alpha=alpha,
                dm=1)

model_dbow.build_vocab(tagged_data)
model_dm.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model_dbow.train(tagged_data,
                total_examples=model_dbow.corpus_count,
                epochs=model_dbow.iter)
    model_dm.train(tagged_data,
                total_examples=model_dm.corpus_count,
                epochs=model_dm.iter)
    # decrease the learning rate
    model_dbow.alpha -= 0.0002
    model_dm.alpha -= 0.0002
    # fix the learning rate, no decay
    model_dbow.min_alpha = model_dbow.alpha
    model_dm.min_alpha = model_dm.alpha

with open( output_file, 'w' ) as outfile:
    for i in range(len(model_dbow.docvecs)):
        mdbowi = model_dbow.docvecs[i].reshape((1,vec_size))
        mdmi = model_dm.docvecs[i].reshape((1,vec_size))
        mi = np.concatenate((mdbowi,mdmi)).reshape((1,2*vec_size))
        np.savetxt(outfile, mi)
