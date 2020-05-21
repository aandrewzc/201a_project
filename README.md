## 201A VLSI Design Automation Project

### Design Rule Matching 

Natural Language Processing Approach

Keywords:
- Glove embedding
- Kazuma embedding
- Fuzzy string matching
- text similarity
- semantic similarity between sentences

Approach
- .rul file yields list of rules:
	```
	rule = dict {
				  'name' = "rule_name"
				  'description' = ["line1", "line2",...]
				  'rule' = ["line1", ...]
				  'layer' = "layer_name"
				  'embedding' = []
				}
	```
- read rules
- sort rules into layers
	* label each rule with a layer (easy for csv hard for rul)
	* match RUL rules to csv counterparts
	* split layer names on LAYER keyword
- match layers between pdks
	* compare lists of layers
	* use embeddings or string matchings
- for each layer
	- match rules between pdks
		* use embeddings or matching techniques

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
