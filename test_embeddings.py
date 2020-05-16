from embeddings import GloveEmbedding, FastTextEmbedding, KazumaCharEmbedding, ConcatEmbedding
import numpy as np

from read_rules import read_rul, read_csv

g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
k = KazumaCharEmbedding()

for w in ['metal1', 'm1', 'METAL1']:
    print('embedding {}'.format(w))
    print(len(g.emb(w)))
    print(len(k.emb(w)))

diff1 = np.array(k.emb('metal1')) - np.array(k.emb('METAL1'))
diff2 = np.array(k.emb('metal1')) - np.array(k.emb('layer'))

# print(np.abs(np.mean(diff1)))
# print(np.abs(np.mean(diff2)))


pdk15_csv = read_csv("calibreDRC_15.csv")
pdk45_csv = read_csv("calibreDRC_45.csv")

pdk15_rul = read_rul("calibreDRC_15.rul")
pdk45_rul = read_rul("calibreDRC_45.rul")

rule1 = pdk15_csv['METAL1 LAYER'][0]
print(rule1)
name, value, description = rule1
sentence = k.emb(description)
print(description.split(' '))

for i in range(len(pdk45_rul)):
    if "Metal1.1" in pdk45_rul[i]:
        print(i, pdk45_rul[i])
        lines = pdk45_rul[i].split('\n')
        print(lines)
        rule = lines[2]
        print(rule)
        words = rule.split(' ')
        print(words)
        sentence = np.array(k.emb(words[0]))
        print(words[0], sentence[0:2])
        N = len(words)
        for i in range(1, N):
            embed = np.array(k.emb(words[i]))
            sentence += embed
            print(words[i], embed[0:2])
        sentence /= N
        print(sentence[0:2])
