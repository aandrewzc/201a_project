from embeddings import GloveEmbedding, FastTextEmbedding, KazumaCharEmbedding, ConcatEmbedding
import numpy as np

g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
k = KazumaCharEmbedding()

for w in ['metal1', 'm1', 'METAL1']:
    print('embedding {}'.format(w))
    print(g.emb(w))
    print(k.emb(w))

diff1 = np.array(k.emb('metal1')) - np.array(k.emb('METAL1'))
diff2 = np.array(k.emb('metal1')) - np.array(k.emb('layer'))

print(np.abs(np.mean(diff1)))
print(np.abs(np.mean(diff2)))
