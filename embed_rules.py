import numpy as np
from embeddings import GloveEmbedding, KazumaCharEmbedding
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
import tensorflow_hub as hub
import os

###############################################################################
# compute_pc() and remove_pc()
# These functions were taken from the SIF embeddings code based on the paper by
# Arora et al.
# source code at https://github.com/PrincetonML/SIF
###############################################################################
def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_
def remove_pc(X,npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


###############################################################################
# def joint_embedding():
# Parameters set1 and set2 are SentenceEmbedding objects.
# Returns weighted sum of embeddings after removing the first principle 
# component of the combined embedding matrices.
###############################################################################
def joint_embedding(set1, set2):
    if set1.features != set2.features:
        print("Embedding must use the same features")
        return None, None

    num_features = len(set1.features)
    N1 = len(set1.pdk)
    N2 = len(set2.pdk)

    partial_embed1 = np.zeros((num_features, N1, set1.size))
    partial_embed2 = np.zeros((num_features, N2, set2.size))

    for i in range(num_features):
        result1 = set1.embed_key(set1.features[i])
        result2 = set2.embed_key(set2.features[i])

        # remove first principle component
        temp = np.row_stack((result1,result2))
        emb = remove_pc(temp, 1)
        partial_embed1[i] = emb[0:N1,:]
        partial_embed2[i] = emb[N1:,:]

    # compute weight sum of embeddings (f[1]*w[1] + f[2]*w[2])
    embed1 = np.tensordot(partial_embed1, set1.weights, axes=(0,0))
    embed2 = np.tensordot(partial_embed2, set2.weights, axes=(0,0))

    return [embed1, embed2]


class SentenceEmbedding:
    def __init__(self, embedding_type, inputs):
        if embedding_type == "char":
            self.wordEmbed = KazumaCharEmbedding()
            self.sentenceEmbed = self.embed_sentence
            self.size = 100
        # elif embedding_type == "glove":
        #     self.wordEmbed = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
        #     self.sentenceEmbed = self.embed_sentence
        #     self.size = 300
        elif embedding_type == "bert":
            try:
                bertEmbed = SentenceTransformer('./bert-base-nli-mean-tokens')
            except OSError as e:
                print(e)
                print("Could not find model in current directory: %s" % os.getcwd())
                exit(1)
            self.sentenceEmbed = bertEmbed.encode
            self.size = 768

        elif embedding_type == "bert-stsb":
            try:
                bertEmbed = SentenceTransformer('./bert-base-nli-stsb-mean-tokens')
            except OSError as e:
                print(e)
                print("Could not find model in current directory: %s" % os.getcwd())
                exit(1)
            self.sentenceEmbed = bertEmbed.encode
            self.size = 768

        elif embedding_type == "universal":
            try:
                univEmbed = hub.load("./universal-sentence-encoder_4")
            except OSError as e:
                print(e)
                print("Could not find model in current directory: %s" % os.getcwd())
                exit(1)
            self.sentenceEmbed = univEmbed
            self.size = 512
        
        else:
            print("Error: Embedding type \"%s\" not recognized" % embedding_type)
            print("Supported types: \"char\", \"bert\", \"bert-stsb\", \"universal\"")
            exit(1)

        self.type = embedding_type
        self.pdk = inputs['pdk']
        self.features = inputs['features']
        self.weights = inputs['weights']
        self.word_counts = inputs['word_counts']
        self.a = inputs['a']
        self.number_replacement = inputs['number_replacement']
        self.remove_pc = inputs['remove_pc']
        self.weigh_capitals = 2


    ###############################################################################
    # embed_sentence():
    # Returns list of embeddings for the provided sentences
    # If self.word_counts != None, computes a weighted average of the word embeddings
    ###############################################################################
    def embed_sentence(self, text):
        embeddings = []
        N = len(text)
        for i in range(N):
            sentence = text[i]
            words = sentence.split(' ')
            num_words = len(words)
            total = np.zeros(self.size)

            for i in range(num_words):
                w = words[i].strip()

                # remove numbers
                if self.number_replacement and w.replace('.','',1).isdigit():
                    w = self.number_replacement

                embed = np.array(self.wordEmbed.emb(w))
                if None in embed:
                    print(w)
                    continue

                # add weight to words that are all caps
                if self.weigh_capitals and w.isalpha() and w.isupper():
                    embed = self.weigh_capitals * embed

                # weigh words based on inverse of probability
                if self.word_counts and w in self.word_counts.keys():
                    prob = self.word_counts[w] / self.word_counts['total-words']
                    weight = self.a / (self.a + prob)
                    embed = weight * embed

                total += embed
                
            result = total / num_words
            embeddings.append(result)
        return embeddings


    ###############################################################################
    # embed_key():
    # Returns a matrix of sentence embeddings for the designated rule feature. 
    # This can be "rule", "description", layer, name, etc.
    # Embedding type is set by self.embedding_type
    ###############################################################################
    def embed_key(self, key):
        pdk = self.pdk
        N = len(pdk)

        sentences = []
        for i in range(N):
            # in case we embed a feature like name, which is not a list
            if isinstance(pdk[i][key], list):
                s = ' '.join(pdk[i][key])
            else: 
                s = pdk[i][key]
            sentences.append(s)

        result = np.array(self.sentenceEmbed(sentences))
        return result

    ###############################################################################
    # embed_all():
    # Compute rule embeddings using a weighted sum of the features.
    # Weights are stored in self.weights and features are stored in self.features.
    # Remove first principle component if self.useSIF == True
    ###############################################################################
    def embed_all(self):
        num_features = len(self.features)
        N = len(self.pdk)
        
        partial_embed = np.zeros((num_features, N, self.size))

        for i in range(num_features):
            result = self.embed_key(self.features[i])

            # remove first principle component
            if self.remove_pc:
                emb = remove_pc(result, 1)
                partial_embed[i] = emb
            else:
                partial_embed [i] = result

        # compute weight sum of embeddings (f[1]*w[1] + f[2]*w[2])
        output = np.tensordot(partial_embed, self.weights, axes=(0,0))
        return output
