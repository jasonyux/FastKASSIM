import logging
import urllib.request
import zipfile
import os
import shutil
import pathlib

from .cassim.Cassim import Cassim

class FastCassim(Cassim):
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
		if metric < FastCassim.LTK or metric > FastCassim.FTK:
			raise Exception(f"""
			Please specify metric to be:
				FastCassim.LTK or {FastCassim.LTK};
				FastCassim.FTK or {FastCassim.FTK};
			""")
		# filter accepted params
		conf_params = {}
		for k,v in default_params.items():
			if params.get(k) is None:
				logging.info(f"param {k}={v} will be used")
				conf_params[k] = v
			else:
				conf_params[k] = params[k]
		conf_params["use_new_delta"] = (metric == FastCassim.LTK)

		logging.info(f"FastCassim Configued mode={metric}, param={conf_params}")
		return metric, conf_params

	def compute_similarity(self, doc1, doc2):
		return self.syntax_similarity_two_documents(doc1, doc2, **self.__params)

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