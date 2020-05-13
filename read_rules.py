import os
import csv

def read_csv(rule_file):
    rules = dict()
    layer = ""
    rules["name"] = rule_file.split('.')[0]

    # open csv file for rule extraction
    with open(rule_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # read file line by line
        for row in reader:

            # lines defining layer names have only one value
            if row[1] == "":
                # sort rules in a dict by their layer
                layer = row[0]
                rules[layer] = []

            # add each rule to the current layer
            elif layer in rules.keys():
                rules[layer].append(row)
    
    # print results
    debug = False
    if (debug):
        for key in rules.keys():
            if key == "name":
                print("*******************************************")
                print(rules[key])
                print("*******************************************")
            else:
                print(key)
                for rule in rules[key]:
                    print("\t %s, %s, %s" % (rule[0], rule[1], rule[2]))

    return rules


def read_rul(rule_file):
    rules = []
    curr_rule = ""
    build_rule = False

    # open file and read line by line
    with open(rule_file) as f:
        for line in f:
            # ignore comments
            if "//" in line:
                continue

            # '{' indicates start of a rule
            if "{" in line:
                build_rule = True
            
            if build_rule:
                curr_rule += line

            # '}' indicates the end of a rule
            if "}" in line:
                build_rule = False
                rules.append(curr_rule)
                curr_rule = ""

    return rules



if __name__ == "__main__":
    csv_files = [f for f in os.listdir('.') if ".csv" in f]
    rul_files = [f for f in os.listdir('.') if ".rul" in f]

    pdk15_csv = read_csv("calibreDRC_15.csv")
    pdk45_csv = read_csv("calibreDRC_45.csv")

    pdk15_rul = read_rul("calibreDRC_15.rul")
    pdk45_rul = read_rul("calibreDRC_45.rul")

    print(pdk45_rul[0].strip().split('{'))
    print(pdk15_rul[0].strip())

    for key in pdk45_csv.keys():
        print(key)

    for key in pdk15_csv.keys():
        print(key)