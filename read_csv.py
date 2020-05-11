import os
import csv

def read_csv(search_dir='.'):
    # get .csv files
    csv_files = [f for f in os.listdir(search_dir) if ".csv" in f]

    rule_set = []

    for rule_file in csv_files:
        rule_list = dict()
        layer = ""
        rule_list["name"] = rule_file.split('.')[0]

        with open(rule_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if row[1] == "":
                    layer = row[0]
                    rule_list[layer] = []
                elif layer in rule_list.keys():
                    rule_list[layer].append(row)
        
        rule_set.append(rule_list)

    debug = False
    if (debug):
        for rule_list in rule_set:
            for key in rule_list.keys():
                if key == "name":
                    print("*******************************************")
                    print(rule_list[key])
                    print("*******************************************")
                else:
                    print(key)
                    for rule in rule_list[key]:
                        print("\t %s, %s, %s" % (rule[0], rule[1], rule[2]))

    pdk15_csv = dict()
    pdk45_csv = dict()

    for rule_list in rule_set:
        name =  rule_list["name"]
        if "15" in name:
            pdk15_csv = rule_list
        elif "45" in name:
            pdk45_csv = rule_list

    return [pdk15_csv, pdk45_csv]


def read_rul(search_dir='.'):
    rul_files = [f for f in os.listdir(search_dir) if ".rul" in f]

    for filename in rul_files:
        rules = []

        curr_rule = ""
        build_rule = False

        print(filename)
        with open(filename) as f:
            for line in f:
                if "//" in line:
                    continue

                if "{" in line:
                    build_rule = True
                
                if build_rule:
                    curr_rule += line

                if "}" in line:
                    build_rule = False
                    rules.append(curr_rule)
                    curr_rule = ""

        print(len(rules))



if __name__ == "__main__":
    set1, set2 = read_csv()
    print(set1['name'])
    total = 0
    for key in set1.keys():
        total += len(set1[key])
    print(total)

    print(set2['name'])
    total = 0
    for key in set2.keys():
        total += len(set2[key])
    print(total)

    read_rul()