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
    """CASSIM: https://github.com/USC-CSSL/CASSIM
    """
    def __init__(self, swbd=False):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        os.environ['STANFORD_PARSER'] = f'{root}/jars/stanford-parser.jar'
        os.environ['STANFORD_MODELS'] = f'{root}/jars/stanford-parser-3.5.2-models.jar'
        if swbd == False:
            self.parser = stanford.StanfordParser(model_path=f"{root}/jars/englishPCFG.ser.gz")
        else:
            self.parser = stanford.StanfordParser(model_path=f"{root}/jars/englishPCFG_swbd.ser.gz")