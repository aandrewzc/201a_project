# 201A VLSI Design Automation Project

## Design Rule Matching 

Natural Language Processing Approach \
Andrew Chen \
004577818

## Setup
This program was tested using python 3.7.7 \
To install the necessary python packages, run:
```
pip install -r requirements.txt
```

## Usage
To perform rule matching, use the following command.        
```
python match_rules.py --pdk1=calibreDRC_45.rul --pdk2=calibreDRC_15.rul
```
By default the rule files used are ./calibre_45.rul and ./calibre_15.rul, included in this tarball submission. To match different PDK rulesets, pass their paths as command line parameters "pdk1" and "pdk2".



### Approach
- Use .rul file to produce a list of rules. Each rule is a dictionary; for example:
	```
	rule[0] = {	
			'name' = "Well.1", 
			'description' = ["Nwell and Pwell must not overlap"],
			'rule' = ["AND nwell pwell"],
			'layer' = "Well",	
		}
	```
- Generate rule embeddings

Keywords:
- Glove embedding
- Kazuma embedding
- Fuzzy string matching
- text similarity
- semantic similarity between sentences

Ideas
- Embeddings
	* sentence embedding weighted sum using word probabilities
	* include name, plain text, SVRF embeddings
	* concatenate word and character embeddings
	* ignore number values in embeddings (no information added)
	* element wise maximum (max pooling)
- Distance Calculation
	* use Euclidean distance and cosine similarity
	* element wise subtraction or multiplication then aggregate


Pre-Trained Models
- Embeddings
	* save path: ~/.embeddings
- Universal Sentence Embeddings
	* tensorflow_hub.load()
- Sentence Transformers
	* save path: ~/.cache/torch/sentence_transformers