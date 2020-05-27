import os
import csv
import re

def read_csv(rule_file):
    rules = []
    layers = []

    curr_rule = dict() 
    curr_layer = ""
    description = ""
    value = ""

    # open csv file for rule extraction
    with open(rule_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # read file line by line
        for row in reader:

            # lines defining layer names have only one value
            if row[1] == "":
                curr_layer = row[0].split('LAYER')[0].strip()
                layers.append(curr_layer)

            # add each rule to the current layer
            elif curr_layer in layers:
                curr_rule['name'] = row[0]
                curr_rule['value'] = row[1]
                curr_rule['description'] = row[2]
                curr_rule['layer'] = curr_layer
                rules.append(curr_rule)
                curr_rule = dict() 

    return rules, layers


def read_rul(rule_file, number_replacement, count_words=False):
    rules = []
    curr_rule = dict()
    description = []
    value = []
    build_rule = False

    if count_words:
        word_counts = dict()
        total_words = 0

    # open file and read line by line
    with open(rule_file) as f:
        for line in f:
            temp = line.strip()

            # ignore comments
            if not temp or temp[0:2] == "//":
                continue

            temp = temp.split('//')[0].strip()  #in case of same line comments of form "rule_name{ //comment"

            # '{' indicates start of a rule
            if temp[-1] == "{":
                build_rule = True
                name = line.split('{')[0].strip()  # extract rule name
                curr_rule['name'] = name

                # set the layer name, assuming format of: "Metal1.2", "RULE_M1002", or "RULE_MIS01"
                # layer is before a dot, before a three digit rule number, or before a two digit rule number
                if "." in name:
                    curr_rule['layer'] = name.split('.')[0]
                elif "_" in name:
                    temp = name.split('_')[1]
                    rule_num = re.search('[0-9][0-9][0-9][A-Za-z]*$', temp)
                    if rule_num:
                        temp2 = rule_num.group(0)
                        curr_rule['layer'] = temp.split(temp2)[0]
                    else:
                        rule_num = re.search('[0-9][0-9][A-Za-z]*$', temp)
                        if rule_num:
                            temp2 = rule_num.group(0)
                            curr_rule['layer'] = temp.split(temp2)[0]
                # otherwise don't set a layer

            # '}' indicates the end of a rule
            elif temp[0] == "}":
                build_rule = False

                # save current rule
                curr_rule['rule'] = value
                curr_rule['description'] = description
                rules.append(curr_rule)
                
                # initialize the next rule
                curr_rule = dict()
                value = []
                description = []

            elif build_rule:
                if "@" in line:
                    x = line.split('@')[1].strip()  # description is text after the @ symbol
                    description.append(x) 
                else:
                    x = line.strip()  # get rule value
                    value.append(x)

                if count_words:
                    words = x.split(' ')
                    total_words += len(words)
                    for w in words:
                        if number_replacement and w.replace('.','',1).isdigit():
                            w = number_replacement

                        if w in word_counts.keys():
                            word_counts[w] += 1
                        else:
                            word_counts[w] = 1

    if count_words:
        word_counts['total-words'] = total_words
        output = (rules, word_counts)
    else:
        output = (rules, None)

    return output



if __name__ == "__main__":
    csv_files = [f for f in os.listdir('.') if ".csv" in f]
    rul_files = [f for f in os.listdir('.') if ".rul" in f]

    pdk15_csv = read_csv("calibreDRC_15.csv")
    pdk45_csv = read_csv("calibreDRC_45.csv")

    pdk15_rul = read_rul("calibreDRC_15.rul")
    pdk45_rul = read_rul("calibreDRC_45.rul")

    for key in pdk45_csv.keys():
        print(key)

    for key in pdk15_csv.keys():
        print(key)
    
    print(pdk45_rul[2])
    print(pdk15_rul[0])
