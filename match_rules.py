import numpy as np
import csv

from sklearn.decomposition import TruncatedSVD
from embeddings import GloveEmbedding, KazumaCharEmbedding
from fuzzywuzzy import fuzz, process
from read_rules import read_csv, read_rul

# Global Variables
NUMBER_REPLACEMENT = "NUMBER"
EMBEDDING_SIZE = 100
THRESHOLD = 0.25
WEIGHTS = [0.5, 1]
features = ['rule', 'description']
num_features = len(features)

weighted_avg = True
use_sif = True

g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
k = KazumaCharEmbedding()
a = 0.001


###############################################################################
# similarity():
# Returns the cosine similarity between two vectors
###############################################################################
def similarity(s1, s2):
    # euclidean = np.linalg.norm(s1-s2)
    cos_dist = 1 - ( np.dot(s1,s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)) )
    return cos_dist


###############################################################################
# embed_sentence():
# Returns an embedding for the provided sentence
# If word_counts != None, computes a weighted average of the word embeddings
###############################################################################
def embed_sentence(sentence, word_counts, replace_nums):
    words = sentence.split(' ')
    num_words = len(words)
    total = np.zeros(EMBEDDING_SIZE)

    for i in range(num_words):
        w = words[i].strip()

        # remove numbers
        if replace_nums and w.replace('.','',1).isdigit():
            w = NUMBER_REPLACEMENT
            print(w)

        char_embed = k.emb(w)
        embed = np.array(char_embed)
        
        if word_counts and w in word_counts.keys():
            prob = word_counts[w]/word_counts['total-words']
            weight = a / (a+prob)
            embed = weight*embed

        total += embed
        
    result = total / num_words
    return result


###############################################################################
# compute_pc() and remove_pc()
# These functions were taken from the SIF embeddings code based on the paper by
# Arora et al.
# source code at https://github.com/PrincetonML/SIF
###############################################################################
def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_
def remove_pc(X,npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


###############################################################################
# embed_key():
# Returns a matrix of sentence embeddings for the designated key
# Parameters:
#   - pdk: list of rules, each rule is dictionary with rule text
#   - key: the specific rule feature to embed
#   - word_count: dictionary of word counts in the original file
#   - replace_nums: flag designating if numbers should be ignored
###############################################################################
def embed_key(pdk, key, word_count, replace_nums=True):
    N = len(pdk)

    result = np.zeros((N, EMBEDDING_SIZE))
    for i in range(N):
        # in case we embed a feature like name, which is not a list
        if isinstance(pdk[i][key], list):
            s = ' '.join(pdk[i][key])
        else: 
            s = pdk[i][key]
        vec_rule = embed_sentence(s, word_count, replace_nums)
        result[i,:] = vec_rule

    return result


###############################################################################
# match_rules():
# Returns a dictionary matching rules from pdk1 to pdk2. 
#   - key: rule name from pdk1
#   - value: list of matching pdk2 rule names
# A rule may be matched to zero or more rules.
###############################################################################
def match_rules(embed1, pdk1, embed2, pdk2):
    N1 = embed1.shape[0]
    N2 = embed2.shape[0]

    matches = dict()
    for i in range(N1):
        s = np.zeros(N2)
        matches[pdk1[i]['name']] = []
        # print(pdk1[i]['name'])

        for j in range(N2):
            s[j] = similarity(embed1[i,:], embed2[j,:])
            if s[j] > THRESHOLD:
                matches[pdk1[i]['name']].append(pdk2[j]['name'])

        # index = np.argmax(s)
        # matches[pdk1[i]['name']] = pdk2[index]['name']
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
    print("----------Parameters----------")
    print("NUMBER_REPLACEMENT: %s" % NUMBER_REPLACEMENT)
    print("EMBEDDING_SIZE = %d" % EMBEDDING_SIZE)
    print("THRESHOLD = %.3f" % THRESHOLD)
    print("WEIGHTS:")
    for i, w in enumerate(WEIGHTS):
        print("\t'%s' = %d" % (features[i], w))
    print("------------------------------")

    # import rules
    print("1. Reading rules...")
    pdk1_csv, layers1 = read_csv("calibreDRC_15.csv")
    pdk2_csv, layers2 = read_csv("calibreDRC_45.csv")
    pdk1_rul, word_count1 = read_rul("calibreDRC_15.rul", NUMBER_REPLACEMENT, weighted_avg)
    pdk2_rul, word_count2 = read_rul("calibreDRC_45.rul", NUMBER_REPLACEMENT, weighted_avg)

    N1 = len(pdk1_rul)
    N2 = len(pdk2_rul)

    # pairs = pair_layers(layers1, layers2, 50)
    # print(pairs)

    # add_csv_data(pdk1_rul, pdk1_csv, 'csv-rul-matchings-15.csv')
    add_csv_data(pdk2_rul, pdk2_csv, word_count2, 'csv-rul-matchings-45.csv')

    return

    # generate and store rule embeddings
    print("2. Generating rule embeddings...")
    partial_embed1 = np.zeros((num_features, N1, EMBEDDING_SIZE))
    partial_embed2 = np.zeros((num_features, N2, EMBEDDING_SIZE))

    for i in range(num_features):
        result1 = embed_key(pdk1_rul, features[i], word_count1)
        result2 = embed_key(pdk2_rul, features[i], word_count2)

        # remove first principle component
        if use_sif:
            temp = np.row_stack((result1,result2))
            emb = remove_pc(temp, 1)
            partial_embed1[i] = emb[0:N1,:]
            partial_embed2[i] = emb[N1:,:]

    # compute weight sum of embeddings (f[1]*w[1] + f[2]*w[2])
    embed1 = np.tensordot(partial_embed1, WEIGHTS, axes=(0,0))
    embed2 = np.tensordot(partial_embed2, WEIGHTS, axes=(0,0))

    # match rules
    print("3. Matching pdk15 to pdk45...")
    matches = match_rules(embed1, pdk1_rul, embed2, pdk2_rul)

    score = check_matches(matches, "15to45.csv")
    print("  %.3f%% correct" % (score*100))

    print("4. Matching pdk45 to pdk15...")
    matches = match_rules(embed2, pdk2_rul, embed1, pdk1_rul)

    score = check_matches(matches, "45to15.csv")
    print("  %.3f%% correct" % (score*100))


if __name__ == "__main__":
    main()