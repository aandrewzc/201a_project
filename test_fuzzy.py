from fuzzywuzzy import fuzz
from fuzzywuzzy import process

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

layers = ['NW', 'ACTIVE', 'METAL1', 'MINTn', 'VSMGn', 'MSMGn', 'MGn', 'VGn', 'GATE', 'VTL/VTH']
scorer = fuzz.ratio
print(scorer)
print(process.extract("METALINT", layers, limit=4, scorer=scorer))
print(process.extract("METALSMG", layers, limit=4, scorer=scorer))
print(process.extract("METAL1", layers, limit=4, scorer=scorer))
print(process.extract("METALG", layers, limit=4, scorer=scorer))
print(process.extract("ACTIVE", layers, limit=4, scorer=scorer))
print(process.extract("VIA1", layers, limit=4, scorer=scorer))
print(process.extract("VIA[2-3]", layers, limit=4, scorer=scorer))
