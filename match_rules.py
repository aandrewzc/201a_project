import numpy as np
import csv

from embeddings import GloveEmbedding, KazumaCharEmbedding
from fuzzywuzzy import fuzz, process
from read_rules import read_csv, read_rul

# global variables
NUMBER_REPLACEMENT = "number-value"
EMBEDDING_SIZE = 100

g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
k = KazumaCharEmbedding()
a = 0.001


def similarity(s1, s2):
    # euclidean = np.linalg.norm(s1-s2)
    cos_sim = np.dot(s1,s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
    return cos_sim


def embed_sentence(sentence, word_counts=None):
    words = sentence.split(' ')
    num_words = len(words)
    total = np.zeros(EMBEDDING_SIZE)

    for i in range(num_words):
        w = words[i].strip()

        # remove numbers
        if w.replace('.','',1).isdigit():
            w = NUMBER_REPLACEMENT

        embed = np.array( k.emb(w) )
        if word_counts:
            prob = word_counts[w]/word_counts['total-words']
            weight = a / (a+prob)
            embed = weight*embed

        total += embed
        
    result = total / num_words
    return result


def embed_rules(pdk, word_counts=None):
    N = len(pdk)
    w1 = 1
    w2 = 1

    result = np.zeros((N, EMBEDDING_SIZE))
    for i in range(N):
        vec_rule = embed_sentence(' '.join(pdk[i]['rule']), word_counts)
        vec_desc = embed_sentence(' '.join(pdk[i]['description']), word_counts)
        result[i,:] = w1*vec_rule + w2*vec_desc 
    return result


def match_rules(embed1, pdk1, embed2, pdk2):
    N1 = embed1.shape[0]
    N2 = embed2.shape[0]

    matches = dict()
    for i in range(N1):
        s = np.zeros(N2)
        for j in range(N2):
            s[j] = similarity(embed1[i,:], embed2[j,:])
            # if s[i] > 0.95:
            #     print(pdk2_rul[i]['rule'], s[i])

        index = np.argmax(s)
        matches[pdk1[i]['name']] = pdk2[index]['name']
        # print(pdk1[i]['name'], pdk2[index]['name'])

    return matches


def pair_layers(list1, list2, threshold):
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


def add_csv_data(rul_file, csv_file, ground_truth):
    rul_count = len(rul_file)
    csv_count = len(csv_file)

    csv_names = [ csv_file[i]['name'] for i in range(csv_count) ]
    csv_values = [ csv_file[i]['value'] for i in range(csv_count) ]
    csv_descrip = [ csv_file[i]['description'] for i in range(csv_count) ]

    manual_matches = []
    with open(ground_truth, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            manual_matches.append( row )

    correct = 0
    right_layer = 0
    # with open('csv-rul-matchings-2.csv', 'w') as f:
    for i in range(rul_count):
        rul_name = rul_file[i]['name']
        rul_descrip = '\n'.join(rul_file[i]['description'])

        # result = process.extractOne(rul_name, csv_names, scorer=fuzz.ratio)
        # index = csv_names.index(result[0])
        # result = process.extractOne(rul_descrip, csv_descrip, scorer=fuzz.ratio)
        # index = csv_descrip.index(result[0])

        scores = []
        w1 = 1
        w2 = 1
        w3 = 0.5
        for j in range(csv_count):
            name_score = fuzz.ratio(rul_name, csv_names[j])
            desc_score = fuzz.ratio(rul_descrip, csv_descrip[j])
            val_score = fuzz.ratio(rul_descrip, csv_values[j])
            total = w1*name_score + w2*desc_score + w3*val_score
            scores.append(total)
        index = np.argmax(np.array(scores))

        n = csv_file[index]['name']
        layer = csv_file[index]['layer']

        if manual_matches[i][2] != layer:
            # print('**' + rul_name + ', ' + n + ', ' + layer)
            right_layer += 0    
        elif manual_matches[i][1] != csv_names[index]:
            # print('-' + rul_name + ', ' + n + ', ' + layer)
            right_layer += 1
        else:
            correct += 1
            right_layer += 1

        rul_file[i]['layer'] = layer
    
    print( 'totally corrrect:', (correct/rul_count)*100 )    
    print( 'right layer:', (right_layer/rul_count)*100 )

    return


def check_matches(matches, match_file):
    total = 0
    n = len(matches)
    with open(match_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            key = row[0]
            values = row[1:]
            if matches[key] in values:
                total += 1
                print(key, matches[key])
    
    return total/n


def main():
    weighted_avg = False

    # import rules
    print("reading rules...")
    pdk1_csv, layers1 = read_csv("calibreDRC_15.csv")
    pdk2_csv, layers2 = read_csv("calibreDRC_45.csv")
    pdk1_rul, word_count1 = read_rul("calibreDRC_15.rul", weighted_avg)
    pdk2_rul, word_count2 = read_rul("calibreDRC_45.rul", weighted_avg)
    print("rules stored.")

    N1 = len(pdk1_rul)
    N2 = len(pdk2_rul)

    # pairs = pair_layers(layers1, layers2, 50)
    # print(pairs)

    # add_csv_data(pdk1_rul, pdk1_csv, 'csv-rul-matchings-1.csv')
    # add_csv_data(pdk2_rul, pdk2_csv, 'csv-rul-matchings-2.csv')

    # generate and store rule embeddings
    print("generating rule embeddings...")
    embed1 = embed_rules(pdk1_rul, word_count1)
    embed2 = embed_rules(pdk2_rul, word_count2)
    print("embeddings stored.")

    # match rules
    print("matching pdk15 to pdk45...")
    matches = match_rules(embed1, pdk1_rul, embed2, pdk2_rul)
    print("rules matched.")

    score = check_matches(matches, "15to45.csv")
    print(score)

    print("matching pdk45 to pdk15...")
    matches = match_rules(embed2, pdk2_rul, embed1, pdk1_rul)
    print("rules matched.")

    score = check_matches(matches, "45to15.csv")
    print(score)



if __name__ == "__main__":
    main()