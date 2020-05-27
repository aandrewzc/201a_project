import numpy as np
from embeddings import GloveEmbedding, KazumaCharEmbedding
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub

class SentenceEmbedding:
    def __init__(self, embedding_type):
        if embedding_type == "char":
            self.wordEmbed = KazumaCharEmbedding()
            self.size = 100
        elif embedding_type == "glove":
            self.wordEmbed = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True)
            self.size = 300
        elif embedding_type == "bert":
            self.bertEmbed = SentenceTransformer('./bert-base-nli-mean-tokens')
            self.size = 768
        elif embedding_type == "bert-stsb":
            self.bertEmbed = SentenceTransformer('./bert-base-nli-stsb-mean-tokens')
            self.size = 768
        elif embedding_type == "universal":
            self.univEmbed = hub.load("./universal-sentence-encoder_4")

        self.type = embedding_type
        self.a = 0.001


    ###############################################################################
    # embed_sentence():
    # Returns an embedding for the provided sentence
    # If word_counts != None, computes a weighted average of the word embeddings
    ###############################################################################
    def embed_sentence(self, sentence, word_counts, number_replacement):
        words = sentence.split(' ')
        num_words = len(words)
        total = np.zeros(self.size)

        for i in range(num_words):
            w = words[i].strip()

            # remove numbers
            if number_replacement and w.replace('.','',1).isdigit():
                w = number_replacement

            embed = np.array(self.wordEmbed.emb(w))
            
            if word_counts and w in word_counts.keys():
                prob = word_counts[w]/word_counts['total-words']
                weight = self.a / (self.a+prob)
                embed = weight*embed

            total += embed
            
        result = total / num_words
        return result


    ###############################################################################
    # embed_key():
    # Returns a matrix of sentence embeddings for the designated key
    # Parameters:
    #   - pdk: list of rules, each rule is dictionary with rule text
    #   - key: the specific rule feature to embed
    #   - word_count: dictionary of word counts in the original file
    #   - replace_nums: flag designating if numbers should be ignored
    ###############################################################################
    def embed_key(self, pdk, key, word_count, number_replacement):
        N = len(pdk)

        result = np.zeros((N, self.size))
        for i in range(N):
            # in case we embed a feature like name, which is not a list
            if isinstance(pdk[i][key], list):
                s = ' '.join(pdk[i][key])
            else: 
                s = pdk[i][key]

            if self.type == "char" or self.type == "glove":
                result[i,:] = self.embed_sentence(s, word_count, number_replacement)

            elif self.type == "bert" or self.type == "bert-stsb":
                result[i,:] = self.bertEmbed.encode(s)

            elif self.type == "universal":
                result[i,:] = self.univEmbed(s)

        return result