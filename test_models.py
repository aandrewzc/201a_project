import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sentences = ["INTERNAL M1 < 0.028", 
             "INTERNAL metal1 < 0.065",
             "INTERNAL M1A < 0.028",
             "INTERNAL M1B < 0.028",
             "EXT V1_M1A_MINT1A < 0.036",
             "EXTERNAL via1 < 0.075",
             "EXT V1_M1B_MINT1B < 0.036"
             ]
sentences = np.array(sentences)

# url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# url="./universal-sentence-encoder_4"
# model1 = hub.load(url)
# universalEncoder = model1(sentences)
# universalEncoder = np.array(universalEncoder)
# print(universalEncoder.shape)


from sentence_transformers import SentenceTransformer

try:
    model2 = SentenceTransformer('./bert-base-nli-mean-tokens')
except:
    print("ERROR: make sure model is downloaded to working directory")
    exit(0)

sentenceBert = model2.encode(sentences)
sentenceBert = np.array(sentenceBert)
print(sentenceBert.shape)

model3 = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
sentenceBert2 = model3.encode(sentences)
sentenceBert2 = np.array(sentenceBert2)
print(sentenceBert2.shape)


from sklearn.cluster import KMeans
num_clusters = 4
cluster_model = KMeans(n_clusters=num_clusters)
cluster_model.fit(sentenceBert)
cluster_assignment = cluster_model.labels_
print(cluster_assignment)

for i in range(num_clusters):
    print("Cluster %d" % (i+1))
    print(sentences[cluster_assignment==i])