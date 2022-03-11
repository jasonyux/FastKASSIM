import logging
import urllib.request
import zipfile
import os
import shutil
import pathlib

from nltk.tree import Tree
from .cassim.Cassim import Cassim

class FastKassim(Cassim):
	LTK = 0
	FTK = 1

	def __init__(self, metric:int, swbd=False, **compute_params):
		super().__init__(swbd=swbd)
		metric, compute_params = self.__configure(metric, compute_params)

		self.__metric = metric
		self.__params = compute_params
		return

	@property
	def metric(self):
		return self.__metric

	@property
	def params(self):
		return self.__params

	def set_metric(self, value):
		self.__metric, self.__params = self.__configure(value, self.__params)
		return
	
	def set_params(self, **value):
		self.__metric, self.__params = self.__configure(self.__metric, value)
		return

	def __configure(self, metric, params):
		default_params = {
			"average": False,
			"sigma": 1,
			"lmbda": 0.4
		}
		if metric < FastKassim.LTK or metric > FastKassim.FTK:
			raise Exception(f"""
			Please specify metric to be:
				FastKassim.LTK or {FastKassim.LTK};
				FastKassim.FTK or {FastKassim.FTK};
			""")
		# filter accepted params
		conf_params = {}
		for k,v in default_params.items():
			if params.get(k) is None:
				logging.info(f"param {k}={v} will be used")
				conf_params[k] = v
			else:
				conf_params[k] = params[k]
		conf_params["use_new_delta"] = (metric == FastKassim.LTK)

		logging.info(f"FastKassim Configued mode={metric}, param={conf_params}")
		return metric, conf_params

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
		return self.syntax_similarity_two_parsed_documents(parsed_doc1, parsed_doc2, **self.__params)

def download():
	# configure folders
	root = pathlib.Path(__file__).parent.resolve()
	target_folder = os.path.join(root, "cassim/jars")
	tmp_folder = os.path.join(root, "tmp")
	pathlib.Path(target_folder).mkdir(parents=False, exist_ok=False)
	pathlib.Path(tmp_folder).mkdir(parents=False, exist_ok=True)

	# download
	print("Downloading https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip")
	fpath, msg = urllib.request.urlretrieve(
		'https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip', 
		os.path.join(tmp_folder, 'stanford_english.jar')
	)

	# extract
	print("Extracting")
	with zipfile.ZipFile(fpath, 'r') as zip_ref:
		zip_ref.extractall(os.path.join(root, "cassim"))
	
	extracted_dir = os.path.join(root, "cassim/stanford-parser-full-2015-04-20")
	pathlib.Path(extracted_dir).rename(target_folder)

	models_dir = os.path.join(target_folder, "stanford-parser-3.5.2-models.jar")
	with zipfile.ZipFile(models_dir, 'r') as zip_ref:
		zip_ref.extractall(target_folder)

	pcfg_file = os.path.join(target_folder, "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
	shutil.copyfile(pcfg_file, os.path.join(target_folder, "englishPCFG.ser.gz"))

	# clean up
	print("Cleaning up")
	shutil.rmtree(tmp_folder)
	print("Done")
	return