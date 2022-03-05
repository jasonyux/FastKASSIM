import logging

from .cassim.Cassim import Cassim


class FastCassim(Cassim):
	NEW_FTK = 0
	OLD_FTK = 1
	CASSIM = 2

	def __init__(self, metric:int, swbd=False, **compute_params):
		super().__init__(swbd=swbd)
		metric, compute_params = self.__configure(metric, compute_params)

		self.__metric = metric
		self.__params = compute_params
		return

	@property
	def metric(self):
		return self.__metric #TODO: maybe also return default params

	@property
	def params(self):
		return self.__params

	def set_metric(self, value):
		self.__metric, self.__params = self.__configure(value, self.__params)
		return
	
	def set_params(self, **value):
		self.__metric, self.__params = self.__configure(self.__metric, value)
		return

	def ____configure(self, metric, params):
		if metric == FastCassim.NEW_FTK:
			params["average"] = False
			params["use_new_delta"] = True
		elif metric == FastCassim.OLD_FTK:
			params["average"] = False
			params["use_new_delta"] = False
		else:
			pass
		return params

	def __configure(self, metric, params):
		params_keys = ["average", "sigma", "lmbda", "use_new_delta"]
		if metric < 0 or metric > 2:
			raise Exception(f"""
			Please specify metric to be:
				FTKCassim.NEW_FTK or {FastCassim.NEW_FTK};
				FTKCassim.OLD_FTK or {FastCassim.OLD_FTK};
				FTKCassim.CASSIM  or {FastCassim.CASSIM}
			""")
		conf_params = {}
		for k,v in params.items():
			if k not in params_keys:
				logging.warning(f"param {k}={v} will not be used")
			else:
				conf_params[k] = v
		
		conf_params = self.____configure(metric, conf_params)
		logging.info(f"FTKCassim Configued mode={metric}, param={conf_params}")
		return metric, conf_params

	def compute_similarity(self, doc1, doc2):
		if self.__metric == FastCassim.NEW_FTK or self.__metric == FastCassim.OLD_FTK:
			return self.ftk_syntax_similarity_two_documents(doc1, doc2, **self.__params)
		else:
			average = self.__params.get("average")
			return self.syntax_similarity_two_documents(doc1, doc2, average=None or average)
