__author__ = 'reihane'
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
    
    def syntax_similarity_two_lists(self, documents1, documents2, average = False): # synax similarity of two lists of documents
        global numnodes
        documents1parsed = []
        documents2parsed = []

        for d1 in range(len(documents1)):
            # print d1
            tempsents = (self.sent_detector.tokenize(documents1[d1].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents1parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents1parsed.append(list(temp))
        for d2 in range(len(documents2)):
            # print d2
            tempsents = (self.sent_detector.tokenize(documents2[d2].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents2parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents2parsed.append(list(temp))
        results ={}
        for d1 in range(len(documents1parsed)):
            # print d1
            for d2 in range(len(documents2parsed)):
                # print d1,d2
                if documents1parsed[d1]=="NA" or documents2parsed[d2] =="NA":
                    # print "skipped"
                    continue
                costMatrix = []
                for i in range(len(documents1parsed[d1])):
                    numnodes = 0
                    tempnode = Node(documents1parsed[d1][i].root().label())
                    new_sentencedoc1 = self.convert_mytree(documents1parsed[d1][i],tempnode)
                    temp_costMatrix = []
                    sen1nodes = numnodes
                    for j in range(len(documents2parsed[d2])):
                        numnodes=0.0
                        tempnode = Node(documents2parsed[d2][j].root().label())
                        new_sentencedoc2 = self.convert_mytree(documents2parsed[d2][j],tempnode)
                        ED = simple_distance(new_sentencedoc1, new_sentencedoc2)
                        ED = ED / (numnodes + sen1nodes)
                        temp_costMatrix.append(ED)
                    costMatrix.append(temp_costMatrix)
                costMatrix = np.array(costMatrix)
                if average==True:
                    return 1-np.mean(costMatrix)
                else:
                    indexes = su.linear_sum_assignment(costMatrix)
                    total = 0
                    rowMarked = [0] * len(documents1parsed[d1])
                    colMarked = [0] * len(documents2parsed[d2])
                    for row, column in indexes:
                        total += costMatrix[row][column]
                        rowMarked[row] = 1
                        colMarked [column] = 1
                    for k in range(len(rowMarked)):
                        if rowMarked[k]==0:
                            total+= np.min(costMatrix[k])
                    for c in range(len(colMarked)):
                        if colMarked[c]==0:
                            total+= np.min(costMatrix[:,c])
                    maxlengraph = max(len(documents1parsed[d1]),len(documents2parsed[d2]))
                    results[(d1,d2)] = 1-total/maxlengraph
        return results

    def syntax_similarity_conversation(self, documents1, average=False): #syntax similarity of each document with its before and after document
        global numnodes
        documents1parsed = []
        for d1 in range(len(documents1)):
            sys.stderr.write(str(d1)+"\n")
            # print documents1[d1]
            tempsents = (self.sent_detector.tokenize(documents1[d1].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents1parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents1parsed.append(list(temp))
        results = OrderedDict()
        for d1 in range(len(documents1parsed)):
            d2 = d1+1
            if d2 == len(documents1parsed):
                break
            if documents1parsed[d1] == "NA" or documents1parsed[d2]=="NA":
                continue
            costMatrix = []
            for i in range(len(documents1parsed[d1])):
                numnodes = 0
                tempnode = Node(documents1parsed[d1][i].root().label())
                new_sentencedoc1 = self.convert_mytree(documents1parsed[d1][i],tempnode)
                temp_costMatrix = []
                sen1nodes = numnodes
                for j in range(len(documents1parsed[d2])):
                    numnodes=0.0
                    tempnode = Node(documents1parsed[d2][j].root().label())
                    new_sentencedoc2 = self.convert_mytree(documents1parsed[d2][j],tempnode)
                    ED = simple_distance(new_sentencedoc1, new_sentencedoc2)
                    ED = ED / (numnodes + sen1nodes)
                    temp_costMatrix.append(ED)
                costMatrix.append(temp_costMatrix)
            costMatrix = np.array(costMatrix)
            if average==True:
                return 1-np.mean(costMatrix)
            else:
                indexes = su.linear_sum_assignment(costMatrix)
                total = 0
                rowMarked = [0] * len(documents1parsed[d1])
                colMarked = [0] * len(documents1parsed[d2])
                for row, column in indexes:
                    total += costMatrix[row][column]
                    rowMarked[row] = 1
                    colMarked [column] = 1
                for k in range(len(rowMarked)):
                    if rowMarked[k]==0:
                        total+= np.min(costMatrix[k])
                for c in range(len(colMarked)):
                    if colMarked[c]==0:
                        total+= np.min(costMatrix[:,c])
                maxlengraph = max(len(documents1parsed[d1]),len(documents1parsed[d2]))
                results[(d1,d2)] = 1-total/maxlengraph#, minWeight/minlengraph, randtotal/lengraph
        return results

    def syntax_similarity_one_list(self, documents1, average): #syntax similarity of each document with all other documents
        global numnodes
        documents1parsed = []
        for d1 in range(len(documents1)):
            #print d1
            tempsents = (self.sent_detector.tokenize(documents1[d1].strip()))
            for s in tempsents:
                if len(s.split())>100:
                    documents1parsed.append("NA")
                    break
            else:
                temp = list(self.parser.raw_parse_sents((tempsents)))
                for i in range(len(temp)):
                    temp[i] = list(temp[i])[0]
                    temp[i] = ParentedTree.convert(temp[i])
                documents1parsed.append(list(temp))
        results ={}
        for d1 in range(len(documents1parsed)):
            #print d1
            for d2 in range(d1+1 , len(documents1parsed)):
                if documents1parsed[d1] == "NA" or documents1parsed[d2]=="NA":
                    continue
                costMatrix = []
                for i in range(len(documents1parsed[d1])):
                    numnodes = 0
                    tempnode = Node(documents1parsed[d1][i].root().label())
                    new_sentencedoc1 = self.convert_mytree(documents1parsed[d1][i],tempnode)
                    temp_costMatrix = []
                    sen1nodes = numnodes
                    for j in range(len(documents1parsed[d2])):
                        numnodes=0.0
                        tempnode = Node(documents1parsed[d2][j].root().label())
                        new_sentencedoc2 = self.convert_mytree(documents1parsed[d2][j],tempnode)
                        ED = simple_distance(new_sentencedoc1, new_sentencedoc2)
                        ED = ED / (numnodes + sen1nodes)
                        temp_costMatrix.append(ED)
                    costMatrix.append(temp_costMatrix)
                costMatrix = np.array(costMatrix)
                if average==True:
                    return 1-np.mean(costMatrix)
                else:
                    indexes = su.linear_sum_assignment(costMatrix)
                    total = 0
                    rowMarked = [0] * len(documents1parsed[d1])
                    colMarked = [0] * len(documents1parsed[d2])
                    for row, column in indexes:
                        total += costMatrix[row][column]
                        rowMarked[row] = 1
                        colMarked [column] = 1
                    for k in range(len(rowMarked)):
                        if rowMarked[k]==0:
                            total+= np.min(costMatrix[k])
                    for c in range(len(colMarked)):
                        if colMarked[c]==0:
                            total+= np.min(costMatrix[:,c])
                    maxlengraph = max(len(documents1parsed[d1]),len(documents1parsed[d2]))
                    results[(d1,d2)] = 1-total/maxlengraph#, minWeight/minlengraph, randtotal/lengraph
        return results
