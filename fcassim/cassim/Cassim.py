__author__ = 'reihane'
from typing import Tuple
import scipy.optimize as su
import numpy as np
import sys
import os
import nltk
import logging
import sys
import pathlib

from nltk.parse import stanford
from nltk.tree import ParentedTree, Tree
from zss import simple_distance, Node
from collections import OrderedDict
from ..ltk.ltk import LabelTreeKernel

numnodes =0
root = pathlib.Path(__file__).parent.resolve()

class Cassim:
    def __init__(self, swbd=False):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        os.environ['STANFORD_PARSER'] = f'{root}/jars/stanford-parser.jar'
        os.environ['STANFORD_MODELS'] = f'{root}/jars/stanford-parser-3.5.2-models.jar'
        if swbd == False:
            self.parser = stanford.StanfordParser(model_path=f"{root}/jars/englishPCFG.ser.gz")
        else:
            self.parser = stanford.StanfordParser(model_path=f"{root}/jars/englishPCFG_swbd.ser.gz")

    def convert_mytree(self, nltktree,pnode):
        global numnodes
        for node in nltktree:
            numnodes+=1
            if type(node) is nltk.ParentedTree:
                tempnode = Node(node.label())
                pnode.addkid(tempnode)
                self.convert_mytree(node,tempnode)
        return pnode

    def syntax_similarity_two_documents(self, doc1, doc2, average=False, sigma=1, lmbda=0.4, use_new_delta=True): #syntax similarity of two single documents
        doc1sents = self.sent_detector.tokenize(doc1.strip())
        doc2sents = self.sent_detector.tokenize(doc2.strip())
        for s1, s2 in zip(doc1sents, doc2sents): # to handle unusual long sentences.
            if len(s1.split()) > 100 or len(s2.split()) > 100:
                logging.info(f"received large documents")
                break
        
        try: #to handle parse errors. Parser errors might happen in cases where there is an unsuall long word in the sentence.
            doc1parsed = self.parser.raw_parse_sents((doc1sents))
            doc2parsed = self.parser.raw_parse_sents((doc2sents))
        except Exception as e:
            sys.stderr.write(str(e))
            return "NA"
        costMatrix = []
        doc1parsed = list(doc1parsed)
        for i in range(len(doc1parsed)):
            doc1parsed[i] = list(doc1parsed[i])[0]
        doc2parsed = list(doc2parsed)
        for i in range(len(doc2parsed)):
            doc2parsed[i] = list(doc2parsed[i])[0]
        for i in range(len(doc1parsed)):
            sentencedoc1 = Tree.convert(doc1parsed[i])
            temp_costMatrix = []
            for j in range(len(doc2parsed)):
                sentencedoc2 = Tree.convert(doc2parsed[j])
                normalized_score = LabelTreeKernel.kernel(sentencedoc1, sentencedoc2, sigma, lmbda, use_new_delta)
                temp_costMatrix.append(normalized_score)
            costMatrix.append(temp_costMatrix)
        costMatrix = np.array(costMatrix)
        if average==True:
            return np.mean(costMatrix)
        else:
            row_ind, col_ind = su.linear_sum_assignment(costMatrix, True)
            total = costMatrix[row_ind, col_ind].sum()
            maxlengraph = max(len(doc1parsed),len(doc2parsed))
            return (total/maxlengraph)

    def syntax_similarity_two_parsed_documents(self, doc1_parsed:"list[Tree]", doc2_parsed:"list[Tree]", average=False, sigma=1, lmbda=0.4, use_new_delta=True):
        costMatrix = []
        for sentencedoc1 in doc1_parsed:
            temp_costMatrix = []
            for sentencedoc2 in doc2_parsed:
                normalized_score = LabelTreeKernel.kernel(sentencedoc1, sentencedoc2, sigma, lmbda, use_new_delta)
                temp_costMatrix.append(normalized_score)
            costMatrix.append(temp_costMatrix)
        costMatrix = np.array(costMatrix)
        if average==True:
            return np.mean(costMatrix)
        else:
            row_ind, col_ind = su.linear_sum_assignment(costMatrix, True)
            total = costMatrix[row_ind, col_ind].sum()
            maxlengraph = max(len(doc1_parsed),len(doc2_parsed))
            return (total/maxlengraph)