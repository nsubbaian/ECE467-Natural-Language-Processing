# Natural Language Processing Project 1
# Spring 2020
# Nithilam Subbaian

## To choose which smoothing function run python NLP_Proj1_Final.py [smoothing function name]
## options for smoothing function name include: "laplace", "JM", "Dir", "AD", "TS"

import nltk
import tqdm
from math import log
import numpy as np
from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
import os
import string
import sys

# Use labeled documents to train
filename_1 = input("Input the Filename that contains the list of labeled training documents: ")
# filename_1 = "corpus1_train.labels"

ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))
bigdoc = dict()
N_doc =0
category_info = dict()
V = []

with tqdm.tqdm(total=os.path.getsize(filename_1)) as pbar:
    with open(filename_1,'r') as f:
        for line in f:
            line_words = line.split()

            line_file = open(line_words[0], 'r').read()
            N_doc+=1
            line_category = line_words[1]

            if line_category not in bigdoc.keys():
                bigdoc[line_category] = dict()
                category_info[line_category] ={}
                category_info[line_category]["documentCount"] = 0
                category_info[line_category]["wordCount"] = 0
            category_info[line_category]["documentCount"] +=1

            tokens = nltk.word_tokenize(line_file)

            for token in tokens:
                token = ps.stem(token)
                if token in string.punctuation:
                    continue
#                 if token in stop_words:
#                     continue
                if token not in bigdoc[line_category].keys():
                    bigdoc[line_category][token] = 1
                    category_info[line_category]["wordCount"] +=1
                else:
                    bigdoc[line_category][token] +=1
                    category_info[line_category]["wordCount"] +=1

                V.append(token)
            pbar.update(len(line))

print("length of N_doc", N_doc)
V_notUnique = V
print("length of V", len(V))
V = np.unique(V)
print("unique length of V", len(V))

logprior = dict()
for c in bigdoc.keys():
    logprior[c] = log(category_info[c]["documentCount"]/N_doc)

# alpha = float(sys.argv[2])

# the best alpha values for each smoothing method
if sys.argv[1] == "laplace":
    alpha = 0.053
elif sys.argv[1] == "JM":
    alpha = 0.01
alpha2 = 100
V_len = len(V)

loglikelihood = dict()
for c in bigdoc.keys():
    loglikelihood[c]= {}
    for word in V:
        if word not in loglikelihood[c].keys():
            loglikelihood[c][word] = {}
        if word not in bigdoc[c].keys():
            bigdoc[c][word] = 0

        count_wc = bigdoc[c][word]

        P_WC = 0
        for c2 in bigdoc.keys():
            if word not in bigdoc[c2].keys():
                add = 0
                P_WC += add
            else:
                add = bigdoc[c2][word]
                P_WC += add
        P_WC = P_WC/len(V)

        # based on arguments, select smoothing function
        if sys.argv[1] == "laplace":
            loglikelihood[c][word] = log((count_wc + alpha)/(category_info[c]["wordCount"] + alpha*V_len))
        elif sys.argv[1] == "JM":
            # Jelinek-Mercer Smoothing
            loglikelihood[c][word] = log((1-alpha)*(count_wc )/(category_info[c]["wordCount"]) + alpha*P_WC)
        elif sys.argv[1] == "Dir":
            # Dirchlet smoothing
            loglikelihood[c][word] = log(((count_wc +alpha*P_WC)/(category_info[c]["wordCount"]+ alpha)))
        elif sys.argv[1] == "AD":
            #  Absolute Discounting smoothing
            loglikelihood[c][word] = log((max(count_wc- alpha, 0) + alpha*len(bigdoc[c].keys())*P_WC)/(category_info[c]["wordCount"]))
        elif sys.argv[1] == "TS":
            # Two-stage smoothing
            loglikelihood[c][word] = log((1-alpha)*(count_wc + alpha2*P_WC)/(category_info[c]["wordCount"]+ alpha2) + alpha*P_WC)

# in training, to evaluate model: divide training set into a smaller training set and a tuning set,
# or use k-fold cross-validation. Do not include documents in both training and test set
# ----> if you tune parameters, improving accuracy as you go, the estimate still might be a bit high

# Use list of unlabeled documents to test
filename_2 = input("Input the Filename that contains the list of unlabeled test documents: ")
# filename_2 = "corpus1_test.list"
output_toFile = []
with tqdm.tqdm(total=os.path.getsize(filename_2)) as pbar:
    with open(filename_2,'r') as f_2:
        for line in f_2:
            fileLocation = line.rstrip("\n\r")

            line_file = open(fileLocation, 'r').read()
            line_file_tokens = nltk.word_tokenize(line_file)

            sum = dict()
            for c in bigdoc.keys():
                sum[c] = logprior[c]
                for pos_i in line_file_tokens:
                    pos_i = ps.stem(pos_i)
                    if pos_i in string.punctuation:
                        continue
#                     if token in stop_words:
#                         continue
                    if pos_i in loglikelihood[c].keys():
                        sum[c] = sum[c] + loglikelihood[c][pos_i]

            output_toFile.append(fileLocation + " " + max(sum, key = sum.get) + "\n")
            pbar.update(len(line))


filename_3 = input("Specify the name of output file to save predictions to: ")
# filename_3 = "corpus1_predictions1.labels"
outfile = open(filename_3, "w")
for output_line in output_toFile:
    outfile.write(output_line)
outfile.close()

# run after to test accuracy of labeled predictions
# perl analyze.pl corpus1_predictions.labels corpus1_test.labels
