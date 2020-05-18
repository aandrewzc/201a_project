import numpy as np
import csv

from embeddings import KazumaCharEmbedding
from fuzzywuzzy import fuzz, process
from read_rules import read_csv, read_rul


def similarity(s1, s2):
    return np.dot(s1,s2) / (np.linalg.norm(s1)*np.linalg.norm(s2))


def embed_sentence(sentence, embedding):
    words = sentence.split(' ')
    num_words = len(words)
    embedding_size = len(embedding.emb(''))  #100
    total = np.zeros(embedding_size)

    for i in range(num_words):
        embed = np.array( embedding.emb(words[i].strip()) )
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


def main():
    k = KazumaCharEmbedding()

    # import rules
    pdk1_csv, layers1 = read_csv("calibreDRC_15.csv")
    pdk2_csv, layers2 = read_csv("calibreDRC_45.csv")
    pdk1_rul = read_rul("calibreDRC_15.rul")
    pdk2_rul = read_rul("calibreDRC_45.rul")

    N1 = len(pdk1_rul)
    N2 = len(pdk2_rul)

    pairs = match_layers(layers1, layers2, 50)
    print(pairs)

    with open('15.csv', 'w') as f:
        names = [ pdk1_rul[i]['name'] for i in range(N1) ]
        for n in names:
            f.write(n+'=\n')

    with open('45.csv', 'w') as f:
        names = [ pdk2_rul[i]['name'] for i in range(N2) ]
        for n in names:
            f.write(n+'=\n')

    add_csv_data(pdk1_rul, pdk1_csv, 'csv-rul-matchings-1.csv')
    # add_csv_data(pdk2_rul, pdk2_csv, 'csv-rul-matchings-2.csv')

    # generate and store rule embeddings
    print("generating rule embeddings...")
    for rule in pdk1_rul:
        rule['embedding'] = embed_sentence('\n'.join(rule['rule']), k)
    for rule in pdk2_rul:
        rule['embedding'] = embed_sentence('\n'.join(rule['rule']), k)
    print("embeddings stored...")

    # e = pdk1_rul[144]['embedding']
    # N = len(pdk2_rul)
    # s = np.zeros(N)
    # for i in range(N):
    #     s[i] = similarity(e, pdk2_rul[i]['embedding'])
    #     if s[i] > 0.95:
    #         print(pdk2_rul[i]['rule'], s[i])

    # index = np.argmax(s)
    # print(pdk1_rul[144]['rule'], pdk2_rul[index]['rule'] )

    # r = pdk2_rul[24]['rule'][0]


if __name__ == "__main__":
    main()