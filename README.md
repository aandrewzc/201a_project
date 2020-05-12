## 201A VLSI Design Automation Project

### Design Rule Matching 

Natural Language Processing Approach

Approach
- CSV = Layer,,
		Rule, value, description
- RUL = Name { 
			@description
			rule
		}

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