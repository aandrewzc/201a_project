import numpy as np

from embeddings import KazumaCharEmbedding
from fuzzywuzzy import fuzz, process
from read_rules import read_csv, read_rul


def embed_sentence(sentence, embedding):
    words = sentence.split(' ')
    num_words = len(words)
    embedding_size = len(embedding.emb(''))  #100
    total = np.zeros(embedding_size)

    for i in range(num_words):
        embed = np.array( embedding.emb(words[i]) )
        total += embed
        
    result = total / num_words
    return result


def match_layers(list1, list2, threshold):
    final_pairs = []
    scorer = fuzz.ratio

    while True:
        pairs = []
        for key in list1:
            result = process.extract(key, list2, limit=2, scorer=scorer)
            match = result[0][0]
            score = result[0][1]
            pairs.append( (key, match, score) )

        max_score = threshold
        perfect_match = False
        best_pair = []
        for pair in pairs:
            key, match, score = pair
            if score == 100:
                perfect_match = True
                list1.remove(key)
                list2.remove(match)
                print("matching", key, match, score)
                final_pairs.append( (key,match) )
            elif score > max_score:
                max_score = score
                best_pair = [key, match]
        
        if not best_pair:
            break
        elif perfect_match:
            continue
        else:
            key, match = best_pair
            list1.remove(key)
            list2.remove(match)
            final_pairs.append( (key,match) )
            print("matching", key, match, max_score)    

    return final_pairs


def main():
    k = KazumaCharEmbedding()

    # import rules
    pdk15_csv = read_csv("calibreDRC_15.csv")
    pdk45_csv = read_csv("calibreDRC_45.csv")
    pdk15_rul = read_rul("calibreDRC_15.rul")
    pdk45_rul = read_rul("calibreDRC_45.rul")

    layers45 = []
    for key in pdk45_csv.keys():
        if key != "name":
            layers45.append(key.split('LAYER')[0])

    layers15 = []
    for key in pdk15_csv.keys():
        if key != "name":
            layers15.append(key.split('LAYER')[0])

    pairs = match_layers(layers15, layers45, 50)
    print(pairs)

    for i in range(len(pdk45_rul)):
        if "Metal1.1" in pdk45_rul[i]:
            print(i, pdk45_rul[i])
            lines = pdk45_rul[i].split('\n')
            print(lines)
            rule = lines[2]
            print(embed_sentence(rule, k))


if __name__ == "__main__":
    main()