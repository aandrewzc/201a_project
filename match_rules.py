import numpy as np
import csv
import sys
import os
import getopt
from tabulate import tabulate

# local files
from src.read_rules import read_csv, read_rul
from src.embed_rules import RuleEmbedding, joint_embedding

###############################################################################
# similarity():
# Returns the cosine similarity between two vectors
###############################################################################
def cosine(s1, s2):
    return 1 - (np.dot(s1,s2) / (np.linalg.norm(s1) * np.linalg.norm(s2)))

def euclidean(s1, s2):
    return np.linalg.norm(s1-s2)


###############################################################################
# match_rules():
# Returns a dictionary matching rules from pdk1 to pdk2. 
#   - key: rule name from pdk1
#   - value: list of matching pdk2 rule names
# A rule may be matched to zero or more rules.
###############################################################################
def match_rules(embed1, pdk1, embed2, pdk2, t, weights):
    N1 = embed1.shape[0]
    N2 = embed2.shape[0]

    matches = dict()
    indices = dict()

    for i in range(N1):
        distances = np.zeros(N2)
        curr = pdk1[i]['name']
        matches[curr] = []
        indices[i] = []

        for j in range(N2):
            d1 = cosine(embed1[i,:], embed2[j,:])
            d2 = euclidean(embed1[i,:], embed2[j,:])
            dist = [d1, d2]
            distances[j] = np.dot(weights, dist)

        # print("mean: %.3f, std: %.3f, min: %.3f, max:%.3f" % (np.mean(distances), np.std(distances), np.min(distances), np.max(distances)))
        mean = np.mean(distances)
        std = np.std(distances)

        threshold = mean - (t*std)

        d_sorted = np.argsort(distances)

        for d in d_sorted:
        # threshold = 10000
        # for j in range(10):
            # d = d_sorted[j]
            match = pdk2[d]['name']

            if distances[d] < threshold:
                matches[curr].append(match)
                indices[i].append(d)
            else:
                # since we sorted distances, loop can end early
                break

    return matches, indices


###############################################################################
# generate_output():
# Writes output to results file.
# Takes dictionary of index matches and dictionaries of pdk rules as inputs.
###############################################################################
def generate_output(filename, matches, pdk1, pdk2, name1, name2):
    headers = ['Layer', name1, name2]

    # csv_output contains the lines to be written to the output file
    csv_output = []
    csv_output.append(','.join(headers))

    # format output for rules that were matched
    table = []
    for index in matches.keys():
        # if no matches were found, report "No match"
        if not matches[index]:
            match = "No match"
            csv_output.append( "%s,%s,No match" % (pdk1[index]['layer'], pdk1[index]['name']) )
        else:
            names = []
            for j in matches[index]:
                names.append(pdk2[j]['name'])
                pdk2[j]['used'] = True
            match = '\n'.join(names)

            csv_output.append( "%s,%s,%s" % (pdk1[index]['layer'], pdk1[index]['name'], names[0]) )
            if len(names) > 1:
                for n in range(1,len(names)):
                    csv_output.append( ",,%s" % names[n] )

        # add entry to the table
        entry = (pdk1[index]['layer'], pdk1[index]['name'], match)
        table.append(entry)
    t = tabulate(table, headers=headers, tablefmt='grid')

    # blank lines to divide two tables
    csv_output.append('')
    csv_output.append('')

    # Headers for unmatched pdk2 rules
    csv_output.append('Layer,%s' % name2)

    # extract rules of pdk2 that were unmatched, sorted by layer
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
        # layer without unmatched rules
        if not unused[layer]:
            entry = (layer, "All matched")
            csv_output.append("%s,All matched" % layer)

        else:
            names = unused[layer]
            entry = (layer, '\n'.join(names))
            csv_output.append( "%s,%s" % (layer, names[0]) )
            if len(names) > 1:
                for n in range(1,len(names)):
                    csv_output.append( ",%s" % names[n] )

        table2.append(entry)
    t2 = tabulate(table2, headers=['Layer', name2], tablefmt='grid')
    
    total_output = '\n'.join(csv_output)
    
    with open(filename, 'w') as f:
        f.write(total_output)

    with open('results.txt', 'w') as f:
        f.write('Matched Rules\n')
        f.write(t)
        f.write('\n\n\nUnmatched Rules\n')
        f.write(t2)


###############################################################################
# pair_layers():
# Reads from a layer config file that stores the matchings of layers between PDKs 
# Returns a list of matchings
###############################################################################
def pair_layers(file):
    if not os.path.exists(file):
        print("Could not find layer config file: %s" % file)
        return None

    final_pairs = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            final_pairs.append(row)

    return final_pairs

###############################################################################
# check_matches():
# This function is only for initially testing, to compare output with manually 
# matched rules.
###############################################################################
def check_matches(matches, match_file):
    correct = 0
    total = 0
    with open(match_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            key = row[0]
            values = row[1:]

            if len(values) == 0 and len(matches[key]) == 0:
                correct += 1
                total += 1
            elif key in matches.keys():
                for match in matches[key]:
                    total += 1
                    if match in values:
                        correct += 1
                        # print(key, matches[key], values)
                        break
    return correct, total


def main():
    output_file = "results.csv"

    # Default parameters, can be overwritten by command line paramters
    replacement = "number"
    threshold = 0.95
    dist_weights = [1, 0]
    feature_weights = [0.25, 1, 0.1]
    features = ['rule', 'description', 'layer']
    num_features = len(features)
    embedding_type = "concat"
    layerfile = "layer_config.csv"

    remove_jointpc = False
    weighted_avg = True
    remove_pc = True
    weigh_capitals = 2
    a = 0.001

    name1 = 'FreePDK45'
    name2 = 'FreePDK15'
    # csv_file1 = "calibreDRC_45.csv"
    # csv_file2 = "calibreDRC_15.csv"
    rul_file1 = "calibreDRC_45.rul"
    rul_file2 = "calibreDRC_15.rul"

    if len(sys.argv) > 1:
        try:
            options = ["number=", "threshold=", "feature_weights=", "type=", 
                        "features=", "dist_weights=", "weighted_avg=", "jointpc=", 
                        "removepc=", "weigh_capitals=", "pdk1=", "pdk2=",
                        "name1=", "name2=", "layer_config="]
            opts, args = getopt.getopt(sys.argv[1:], "", options)
        except getopt.GetoptError as err:
            print(err)
            sys.exit(2)

        for option, arg in opts:
            if option == "--number":
                replacement = arg
            elif option == "--threshold":
                threshold = float(arg)
            elif option == "--feature_weights":
                feature_weights = [float(num) for num in arg.split(',')]
            elif option == "--features":
                features = arg.split(',')
            elif option == "--type":
                embedding_type = arg
            elif option == "--dist_weights":
                dist_weights = [float(num) for num in arg.split(',')]
            elif option == "--weigh_capitals":
                weigh_capitals = float(arg)
            elif option == "--weighted_avg":
                if arg in ["True", "true", 'T', 't']:
                    weighted_avg = True
                else:
                    weighted_avg = False
            elif option == "--jointpc":
                if arg in ["True", "true", 'T', 't']:
                    remove_jointpc = True
                else:
                    remove_jointpc = False
            elif option == "--removepc":
                if arg in ["True", "true", 'T', 't']:
                    remove_pc = True
                else:
                    remove_pc = False

            elif option == "--pdk1":
                rul_file1 = arg
            elif option == "--pdk2":
                rul_file2 = arg
            elif option == "--name1":
                name1 = arg
            elif option == "--name2":
                name2 = arg
            elif option == "--layer_config":
                layerfile = arg

    layer_pairs = pair_layers(layerfile)
    if not layer_pairs:
        threshold *= 3

    print("------------Parameters------------")
    print("embedding type: %s" % embedding_type)
    print("replacement: %s" % replacement)
    print("threshold = %.3f" % threshold)
    print("feature weights:")
    for i, w in enumerate(feature_weights):
        print("  '%s' = %.2f" % (features[i], w))
    print("distance weights:")
    print("  cosine = %.2f\n  euclidean = %.2f" % (dist_weights[0], dist_weights[1]))
    if weigh_capitals:
        print("weighing capitalized words by %.3f" % weigh_capitals)
    if weighted_avg:
        print("using weighted average")
    if remove_pc:
        print("removing first principle component")
    if remove_jointpc:
        print("removing pc of joint matrix")
    print("----------------------------------")

    # import rules
    print("1. Reading rules...")
    # pdk1_csv, layers1 = read_csv(csv_file1)
    # pdk2_csv, layers2 = read_csv(csv_file2)
    pdk1_rul, word_count1 = read_rul(rul_file1, replacement, weighted_avg)
    pdk2_rul, word_count2 = read_rul(rul_file2, replacement, weighted_avg)

    # convert to numpy arrays
    pdk1_rul = np.array(pdk1_rul)
    pdk2_rul = np.array(pdk2_rul)
    print("  %s has %d rules." % (name1, len(pdk1_rul)))
    print("  %s has %d rules." % (name2, len(pdk2_rul)))

    
    # generate and store rule embeddings
    print("2. Generating rule embeddings...")
    input1 = {
        'pdk': pdk1_rul,
        'features': features,
        'weights': feature_weights,
        'word_counts': word_count1,
        'a': a,
        'number_replacement': replacement,
        'remove_pc': remove_pc,
        'weigh_capitals': weigh_capitals
    }

    input2 = {
        'pdk': pdk2_rul,
        'features': features,
        'weights': feature_weights,
        'word_counts': word_count2,
        'a': a,
        'number_replacement': replacement,
        'remove_pc': remove_pc,
        'weigh_capitals': weigh_capitals
    }

    E1 = RuleEmbedding(embedding_type, input1)
    E2 = RuleEmbedding(embedding_type, input2)

    if remove_jointpc:
        embed1, embed2 = joint_embedding(E1,E2)
    else:
        embed1 = E1.embed_all()
        embed2 = E2.embed_all()

    print("3. Matching %s to %s..." % (name1, name2))
    # reads input file containing 1:1 layer matchings, outputs list of tuples
    
    if layer_pairs:
        matches = {}
        names = dict()
        for pair in layer_pairs:
            layer1 = pair[0].strip()
            layer2 = pair[1:]

            # extract indices of rules that belong to the current layers
            set1 = [index for index in range(len(pdk1_rul)) if pdk1_rul[index]['layer'] == layer1]
            set2 = [index for index in range(len(pdk2_rul)) if pdk2_rul[index]['layer'] in [l.strip() for l in layer2]]

            # match only rules belonging to each layer
            reduced_names, reduced_matches = match_rules(embed1[set1], pdk1_rul[set1], embed2[set2], pdk2_rul[set2], threshold, dist_weights)
            names.update(reduced_names)

            # map reduced pdk indicies back to full pdk indicies
            for key in reduced_matches.keys():
                i = set1[key]  # index for the rule in pdk1
                matches[i] = []
                for match in reduced_matches[key]:
                    matches[i].append(set2[match])  # add index of each match in pdk2
    
    else:
        # match rules
        names, matches = match_rules(embed1, pdk1_rul, embed2, pdk2_rul, threshold, dist_weights)

    # correct,total = check_matches(names, "45to15.csv")
    # print("  %d/%d: %.3f%% correct" % (correct,total,correct/total*100))

    print("4. Writing output to %s" % output_file)
    generate_output(output_file, matches, pdk1_rul, pdk2_rul, name1, name2)


if __name__ == "__main__":
    main()