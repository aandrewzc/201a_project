# 201A VLSI Design Automation Project

## Design Rule Matching 

Natural Language Processing Approach \
Andrew Chen

## Setup
This program was created using Python 3.7.7 locally.

Note that the eeapps server seems to only have Python 3.4, which is not sufficient for installing packages like numpy. This should be run using at least Python 3.7 to ensure that everything works properly.

To install all the necessary python packages, run:
```
pip install -r requirements.txt
```

During the first run, the program will download word embedding models for the Glove and character embeddings from the "embeddings" library. These are stored by default in ~/.embeddings/. After these files are downloaded, they can be reused on subsequent runs. 

## Usage
### Defaults
To perform rule matching between FreePDK45 and FreePDK15, use the following command:       
```
python match_rules.py
```
This will read from the files included in this directory: calibreDRC_45.rul,calibreDRC_15.rul, layer_config.csv, along with the source code under src/

### Comparing New PDKs
To match the rules of different PDKs, add commmand line arguments as shown below:
```
python match_rules.py --pdk1="calibreDRC_45.rul" --pdk2="calibreDRC_15.rul" --name1="FreePDK45" --name2="FreePDK15" --layer_config=""
```
The parameters {"pdk1", "pdk2"} are the .rul files, {"name1", "name2"} are the names for each PDK, and "layer_config" is a csv file that matches the layers between the two PDKs. The default values are ...

pdk1 = "./calibreDRC_45.rul" \
pdk2 = "./calibreDRC_15.rul" \
name1 = "FreePDK45" \
name2 = "FreePDK15" \
layer_config = "layer_config.csv"

Notes: 
* The names are only used to label the output, they can take any value.
* Each line of the layer_config file should have a layer from pdk1, followed by a comma, followed by a comma separated list of all the matching layers in pdk2. Example shown below:
	```
	Well,NW
	Active,ACT
	Implant,NIM,PIM
	Metal1,M1
	Via1,V1
	...
	...
	```
* If the layer config file is left blank, or the provided file name does not exist, the rules are compared against all other rules, not just the ones of the corresponding layer.
* Using the wrong layer file when matching rules will likely produce an error in the code and no correct matches.

### Other parameters
These extra command line parameters were added to help test different values. For acutal usage, these are not needed.

*  --type={ embedding type to use: char, glove, concat, universal }
	* default: concat
* --threshold={ float to control threshold of a rule match }
	* default: 0.95
* --number={ string to replace numbers with }
	* default: "number"
* --weigh_capitals={ multiplier used to weigh capitalized words }
	* default: 2
* --weighted_avg={ boolean indicating if weighted average should be used * }
	* default: True
* --removepc={ boolean indicating if first principal component should be * removed }
	* default: True
* --features={ comma separated list of rule features to embed }
	* default: rule,description,layer
* --feature_weights={ comma separated list of weights to use for * combining feature embeddings }
	* default: 0.25,1,0.1
* --dist_weights={ comma separated list of weights to use for combining cosine and euclidean distances } 
	* default: 1,0

## Approach
1. Read input from .rul files. Store a list of rules. Each rule is a dictionary, for example:
	```
	rule[0] = {	
			'name' = "Well.1", 
			'description' = ["Nwell and Pwell must not overlap"],
			'rule' = ["AND nwell pwell"],
			'layer' = "Well",	
		}
	```
2. Generate rule embeddings
	- Uses a weighted average of word embeddings.
	- Each word embedding is the concatenation of character and Glove word embeddings.
3. Match the rules
	- Uses cosine distance to compare each rule of PDK1 with the rules in PDK2.
4. Write output to results.csv
	- This also produces the file results.txt, which has an easy to read table format.
