from embeddings import GloveEmbedding, FastTextEmbedding, KazumaCharEmbedding, ConcatEmbedding

g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
#f = FastTextEmbedding()
k = KazumaCharEmbedding()
#c = ConcatEmbedding([g, k])
for w in ['canada', 'vancouver', 'toronto']:
    print('embedding {}'.format(w))
    print(g.emb(w))
#    print(f.emb(w))
    print(k.emb(w))
#    print(c.emb(w))
