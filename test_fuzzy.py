from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from read_rules import read_csv, read_rul

# ratio
# partial_ratio
# token_sort_ratio
# token_set_ratio

# out = fuzz.partial_ratio("METALINT", "MINTn")
# print(out)

# a = fuzz.partial_ratio("METALSMG", "VSMGn")
# print(a)

# out = fuzz.partial_ratio("METALSMG", "MSMGn")
# print(out)

# out = fuzz.partial_ratio("METALSMG", "MGn")
# print(out)

scorer = fuzz.ratio

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

final_pairs = []
list1 = layers45
list2 = layers15
while True:
    pairs = []
    for key in list1:
        result = process.extract(key, list2, limit=2, scorer=scorer)
        match = result[0][0]
        score = result[0][1]
        pairs.append( (key, match, score) )
        # print(key, ':', match, score)

    max_score = 50
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

print(final_pairs)
