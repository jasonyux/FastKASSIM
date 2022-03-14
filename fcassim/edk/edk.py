import logging

from nltk.tree import Tree


class EditDistanceKernel(object):
	NAME = "EditDistanceKernel"
	EDK = 3

	def __init__(self) -> None:
		pass

	@staticmethod
	def kernel(tree_x:Tree, tree_y:Tree, **params) -> float:
		"""
        returns a normalized edit distance score
        """
		return

	@staticmethod
	def config(metric, **user_configs):
		"""returns customized parameters but relevant to the current kernel

		Returns:
			dict: parameters relevant to EditDistanceKernel
		"""
		default_params = {
			"average": False
		}
		# filter accepted params
		conf_params = {}
		for k,v in default_params.items():
			if user_configs.get(k) is None:
				conf_params[k] = v
			else:
				conf_params[k] = user_configs[k]
		
		return conf_params