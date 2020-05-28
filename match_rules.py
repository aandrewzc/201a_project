import numpy as np
import csv
import sys

from embeddings import GloveEmbedding, KazumaCharEmbedding
from fuzzywuzzy import fuzz, process
from read_rules import read_csv, read_rul
from embed_rules import SentenceEmbedding, joint_embedding
from tabulate import tabulate

###############################################################################
# similarity():
# Returns the cosine similarity between two vectors
###############################################################################
def similarity(s1, s2):
    return np.dot(s1,s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))

def euclidean(s1, s2):
    return np.linalg.norm(s1-s2)


###############################################################################
# match_rules():
# Returns a dictionary matching rules from pdk1 to pdk2. 
#   - key: rule name from pdk1
#   - value: list of matching pdk2 rule names
# A rule may be matched to zero or more rules.
###############################################################################
def match_rules(embed1, pdk1, embed2, pdk2, threshold):
    N1 = embed1.shape[0]
    N2 = embed2.shape[0]

    matches = dict()
    indices = dict()

    for i in range(N1):
        s = np.zeros(N2)
        matches[pdk1[i]['name']] = []
        indices[i] = []

        for j in range(N2):
            s[j] = similarity(embed1[i,:], embed2[j,:])
            if s[j] > threshold:
                matches[pdk1[i]['name']].append(pdk2[j]['name'])
                indices[i].append(j)

        # index = np.argmax(s)
        # matches[pdk1[i]['name']] = pdk2[index]['name']
        # print(pdk1[i]['name'], pdk2[index]['name'])

    return matches, indices


def generate_output(filename, matches, pdk1, pdk2, name1, name2):
    """
    # generate ground truth from csv
    table = []
    with open('45to15.csv', newline='') as csvfile:
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            if not row[1]:
                m = ''
            else:
                m = '\n'.join(row[1:])
            entry = (row[0].split('.')[0],row[0], m)
            table.append(entry)
    t = tabulate(table, headers=['Layer','PDK45', 'PDK15'],tablefmt='grid')
    """
    headers = ['Layer', name1, name2]
    table = []
    for index in matches.keys():
        if not matches[index]:
            match = "No match"
        else:
            names = []
            for j in matches[index]:
                names.append(pdk2[j]['name'])
                pdk2[j]['used'] = True
            match = '\n'.join(names)

        entry = (pdk1[index]['layer'], pdk1[index]['name'], match)
        table.append(entry)
    t = tabulate(table, headers=headers, tablefmt='grid')

    unused = {}
    for rule in pdk2:
        # store layers
        layer = rule['layer']
        if layer not in unused.keys():
            unused[layer] = []

        # if rule is unused
        if 'used' not in rule.keys():
            unused[layer].append(rule['name'])
    table2 = []
    for layer in unused.keys():
        if not unused[layer]:
            entry = (layer, "All matched")
        else:
            entry = (layer, '\n'.join(unused[layer]))
        table2.append(entry)
    t2 = tabulate(table2, headers=['Layer', name2], tablefmt='grid')

    with open(filename, 'w') as f:
        f.write('Matched Rules\n')
        f.write(t)
        f.write('\n\n\nUnmatched Rules\n')
        f.write(t2)



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


def add_csv_data(rul_file, csv_file, rul_word_count, ground_truth):
    rul_count = len(rul_file)
    csv_count = len(csv_file)

    # csv_names = [ csv_file[i]['name'] for i in range(csv_count) ]
    # csv_values = [ csv_file[i]['value'] for i in range(csv_count) ]
    # csv_descrip = [ csv_file[i]['description'] for i in range(csv_count) ]

    w1 = 0
    w2 = 1

    rul_desc_emb = embed_key(rul_file, 'description', rul_word_count, False)
    rul_name_emb = embed_key(rul_file, 'name', None, False)
    rul_embed = w1*rul_desc_emb + w2*rul_name_emb

    csv_desc_emb = embed_key(csv_file, 'description', None, False)
    csv_name_emb = embed_key(csv_file, 'name', None, False)
    csv_embed = w1*csv_desc_emb + w2*csv_name_emb

    manual_matches = []
    with open(ground_truth, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            manual_matches.append( row )

    correct = 0
    print("Rule, Match, Correct Match")

    for i in range(rul_count):
        # rul_name = rul_file[i]['name']
        # rul_descrip = ' '.join(rul_file[i]['description'])
        min_score = 10000
        index = -1
        for j in range(csv_count):
            s = similarity(rul_embed[i,:], csv_embed[j,:])
            if s < min_score:
                min_score = s
                index = j

        # check accuracy
        name = csv_file[index]['name']
        correct_name = manual_matches[i][1]
        if correct_name != name:
            if correct_name:
                correct_i = [ i for i in range(csv_count) if csv_file[i]['name'] == correct_name][0]
                s = similarity(rul_embed[i,:], csv_embed[correct_i,:])
                print("%s, %s (%.6f), %s (%.6f)" % (rul_file[i]['name'], name, min_score, correct_name, s) )
            else:
                print("%s, %s, %.6f" % (rul_file[i]['name'], name, min_score) )
        else:
            correct += 1

        # rul_file[i]['layer'] = layer
    
    print( 'totally corrrect:', (correct/rul_count)*100 )    
    return


def check_matches(matches, match_file):
    total = 0
    n = len(matches.keys())
    with open(match_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            key = row[0]
            values = row[1:]

            if len(values) == 0 and len(matches[key]) == 0:
                total += 1
                print(key, matches[key])
            else:
                for match in matches[key]:
                    if match in values:
                        total += 1
                        # print(key, matches[key], values)
                        break
    return total/n


def main():
    if len(sys.argv) > 1:
        print(sys.argv[1:])

    NUMBER_REPLACEMENT = "NUMBER"
    THRESHOLD = 0.5
    WEIGHTS = [0.5, 1]
    features = ['rule', 'description']
    num_features = len(features)
    embedding_type = "char"
    output_file = "output.txt"

    remove_jointpc = False
    weighted_avg = True
    remove_pc = True
    a = 0.001

    csv_file1 = "calibreDRC_45.csv"
    csv_file2 = "calibreDRC_15.csv"
    rul_file1 = "calibreDRC_45.rul"
    rul_file2 = "calibreDRC_15.rul"

    print("----------Parameters----------")
    print("NUMBER_REPLACEMENT: %s" % NUMBER_REPLACEMENT)
    print("THRESHOLD = %.3f" % THRESHOLD)
    print("WEIGHTS:")
    for i, w in enumerate(WEIGHTS):
        print("\t'%s' = %.2f" % (features[i], w))
    print("------------------------------")

    # import rules
    print("1. Reading rules...")
    pdk1_csv, layers1 = read_csv(csv_file1)
    pdk2_csv, layers2 = read_csv(csv_file2)
    pdk1_rul, word_count1 = read_rul(rul_file1, NUMBER_REPLACEMENT, weighted_avg)
    pdk2_rul, word_count2 = read_rul(rul_file2, NUMBER_REPLACEMENT, weighted_avg)

    N1 = len(pdk1_rul)
    N2 = len(pdk2_rul)

    # pairs = pair_layers(layers1, layers2, 50)
    # print(pairs)

    # add_csv_data(pdk1_rul, pdk1_csv, 'csv-rul-matchings-15.csv')
    # add_csv_data(pdk2_rul, pdk2_csv, word_count2, 'csv-rul-matchings-45.csv')
    
    # generate and store rule embeddings
    print("2. Generating rule embeddings...")
    input1 = {
        'pdk': pdk1_rul,
        'features': features,
        'weights': WEIGHTS,
        'word_counts': word_count1,
        'a': a,
        'number_replacement': NUMBER_REPLACEMENT,
        'remove_pc': remove_pc
    }

    input2 = {
        'pdk': pdk2_rul,
        'features': features,
        'weights': WEIGHTS,
        'word_counts': word_count2,
        'a': a,
        'number_replacement': NUMBER_REPLACEMENT,
        'remove_pc': remove_pc
    }

    E1 = SentenceEmbedding(embedding_type, input1)
    E2 = SentenceEmbedding(embedding_type, input2)

    if remove_jointpc:
        embed1, embed2 = joint_embedding(E1,E2)
    else:
        embed1 = E1.embed_all()
        embed2 = E2 .embed_all()

    # match rules
    print("3. Matching pdk45 to pdk15...")
    matches1, index1 = match_rules(embed1, pdk1_rul, embed2, pdk2_rul, THRESHOLD)

    score = check_matches(matches1, "45to15.csv")
    print("  %.3f%% correct" % (score*100))

    print("4. Matching pdk15 to pdk45...")
    matches2, index2 = match_rules(embed2, pdk2_rul, embed1, pdk1_rul, THRESHOLD)

    score = check_matches(matches2, "15to45.csv")
    print("  %.3f%% correct" % (score*100))

    print("Writing output to %s" % output_file)
    generate_output(output_file, index1, pdk1_rul, pdk2_rul, 'FreePDK45', 'FreePDK15')


if __name__ == "__main__":
    main()