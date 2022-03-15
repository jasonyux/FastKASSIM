__author__ = 'jasonyux'

import logging
import urllib.request
import zipfile
import os
import shutil
import pathlib

from fkassim.ltk.ltk import LabelTreeKernel
from fkassim.edk.edk import EditDistanceKernel
from fkassim.kassim.Kassim import Kassim

class FastKassim(Kassim):
	LTK = LabelTreeKernel.LTK
	FTK = LabelTreeKernel.FTK
	ED = EditDistanceKernel.EDK

	def __init__(self, metric:int, swbd=False, **compute_params):
		super().__init__(swbd=swbd)
		self.Kernel = None
		
		# this also configures kernel
		metric, compute_params = self.__configure(metric, compute_params)
		self.__metric = metric
		self.kernel_params = compute_params
		return

	@property
	def metric(self):
		return self.__metric

	@property
	def params(self):
		return self.kernel_params

	def set_metric(self, value):
		self.__metric, self.kernel_params = self.__configure(value, self.kernel_params)
		return
	
	def set_params(self, **value):
		self.__metric, self.kernel_params = self.__configure(self.__metric, value)
		return

	def __configure_kernel(self, metric, **params):
		if self.__metric == FastKassim.LTK:
			self.Kernel = LabelTreeKernel
			return LabelTreeKernel.config(metric, **params)
		elif self.__metric == FastKassim.FTK:
			self.Kernel = LabelTreeKernel
			return LabelTreeKernel.config(metric, **params)
		elif self.__metric == FastKassim.ED:
			self.Kernel = EditDistanceKernel
			return EditDistanceKernel.config(metric, **params)
		return

	def __configure(self, metric, params):
		"""
		Main entry point for configuring kernel method, parameters, and etc
		"""
		if metric < FastKassim.LTK or metric > FastKassim.ED:
			raise Exception(f"""
			Please specify metric to be:
				FastKassim.LTK or {FastKassim.LTK};
				FastKassim.FTK or {FastKassim.FTK};
				FastKassim.ED or {FastKassim.ED};
			""")
		# update metric
		self.__metric = metric

		# configure kernel method call AND its parameters
		conf_params = self.__configure_kernel(metric, **params)

		logging.info(f"FastKassim Configued kernel={self.Kernel.NAME}, param={conf_params}")
		return metric, conf_params


def download():
	# configure folders
	root = pathlib.Path(__file__).parent.resolve()
	target_folder = os.path.join(root, "kassim/jars")
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
		zip_ref.extractall(os.path.join(root, "kassim"))
	
	extracted_dir = os.path.join(root, "kassim/stanford-parser-full-2015-04-20")
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