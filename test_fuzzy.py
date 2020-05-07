from fuzzywuzzy import fuzz
from fuzzywuzzy import process

out = fuzz.ratio("this is a test", "this is a test!")
print(out)

out = fuzz.partial_ratio("this is a test", "this is a test!!")
print(out)
