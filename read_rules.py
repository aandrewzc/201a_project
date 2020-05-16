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
    curr_rule = dict()
    description = []
    value = []
    build_rule = False

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
                curr_rule['name'] = line.split('{')[0].strip()  # extract rule name

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
                    description.append( line.split('@')[1].strip() ) # description is text after the @ symbol
                else:
                    value.append( line.strip() )  # get rule value

    return rules



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
