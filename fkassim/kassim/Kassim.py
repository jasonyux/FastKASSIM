__author__ = 'jasonyux'
import os
import nltk
import pathlib
import numpy as np
import scipy.optimize as su

from nltk.parse import stanford
from nltk.tree import Tree

root = pathlib.Path(__file__).parent.resolve()

class Kassim:
	"""
	Much of this code is adapted from CASSIM: https://github.com/USC-CSSL/CASSIM
	As we use tree kernels, and intend for FastKASSIM to be used beyond conversations, we thus use the name KASSIM (Kernel-bAsed Syntactic SIMilarity Metric).
	"""

	def __init__(self, swbd=False, Kernel:object=None, **compute_params):
		self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		os.environ['STANFORD_PARSER'] = f'{root}/jars/stanford-parser.jar'
		os.environ['STANFORD_MODELS'] = f'{root}/jars/stanford-parser-3.5.2-models.jar'
		if swbd == False:
			self.parser = stanford.StanfordParser(model_path=f"{root}/jars/englishPCFG.ser.gz")
		else:
			self.parser = stanford.StanfordParser(model_path=f"{root}/jars/englishPCFG_swbd.ser.gz")
		
		self.kernel_params = compute_params
		self.Kernel = Kernel

	def parse_document(self, doc, tokenizer=None, parser=None):
		"""parses a document (i.e. a collection of sentences)

		Args:
			doc (str): a document consisting of one or more sentences
			tokenizer (Callable, optional): Tokenizer to tokenize the document into sentences. Defaults to self.sent_detector.tokenize.
			parser (Callable, optional): Parser that can parse a list of sentences into trees. Defaults to self.parser.raw_parse_sents.

		Returns:
			list: list of parse tree for each sentence in the document
		"""
		if tokenizer is None:
			tokenizer = self.sent_detector.tokenize
		if parser is None:
			parser = self.parser.raw_parse_sents
		
		parsed_sent = []
		doc_sents = tokenizer(doc.strip())
		try:
			doc_parsed = parser((doc_sents))
		except Exception as e:
			print(e)
			return []
		doc_parsed = list(doc_parsed)
		for i in range(len(doc_parsed)):
			doc_parsed_i = list(doc_parsed[i])[0]
			parsed_sent_i = Tree.convert(doc_parsed_i)
			parsed_sent.append(parsed_sent_i)
		return parsed_sent

	def compute_similarity(self, doc1, doc2):
		"""compute syntax similarity between two documents

		Args:
			doc1 (str): document 1 (i.e. a collection of sentences)
			doc2 (str): document 2 (i.e. a collection of sentences) to compare against

		Returns:
			float: normalized similarity score
		"""
		parsed_doc1 = self.parse_document(doc1)
		parsed_doc2 = self.parse_document(doc2)
		return self.compute_similarity_preparsed(parsed_doc1, parsed_doc2)

	def compute_similarity_preparsed(self, parsed_doc1, parsed_doc2):
		"""compute syntax similarity between two parsed documents

		Args:
			doc1 (list): list of syntax trees for each sentence in your document 1
			doc2 (list): list of syntax trees for each sentence in your document 2

		Returns:
			float: normalized similarity score
		"""
		if len(parsed_doc1) == 0 or len(parsed_doc2) == 0:
			print(f"""
			Error: Empty document passed in. len(parsed_doc1)={len(parsed_doc1)} and len(parsed_doc2)={len(parsed_doc2)}
			""")
			return -1.
		return self.syntax_similarity_two_parsed_documents(parsed_doc1, parsed_doc2, **self.kernel_params)

	def syntax_similarity_two_parsed_documents(self, doc1_parsed:"list[Tree]", doc2_parsed:"list[Tree]", average=False, **kernal_params):
		costMatrix = []
		for sentencedoc1 in doc1_parsed:
			temp_costMatrix = []
			for sentencedoc2 in doc2_parsed:
				normalized_score = self.Kernel.kernel(sentencedoc1, sentencedoc2, **kernal_params)
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
