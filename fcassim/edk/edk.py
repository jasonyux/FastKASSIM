from zss import Node, simple_distance
from nltk.tree import Tree, ParentedTree


class EditDistanceKernel(object):
	NAME = "EditDistanceKernel"
	EDK = 3

	def __init__(self) -> None:
		pass

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

	@staticmethod
	def nltk_to_zss(nltktree:Tree):
		nltktree = ParentedTree.convert(nltktree)
		root_node = Node(nltktree.root().label())
		return EditDistanceKernel.__nltk_to_zss(nltktree, root_node, 0)

	@staticmethod
	def __nltk_to_zss(nltktree:ParentedTree, pnode, numnodes):
		for node in nltktree:
			numnodes += 1
			if type(node) is ParentedTree:
				tempnode = Node(node.label())
				pnode.addkid(tempnode)
				_, numnodes = EditDistanceKernel.__nltk_to_zss(node, tempnode, numnodes)
		return pnode, numnodes

	@staticmethod
	def kernel(tree_x:Tree, tree_y:Tree, **params) -> float:
		"""
		returns a normalized edit distance score
		"""
		tree_x, num_nodes_x = EditDistanceKernel.nltk_to_zss(tree_x)
		tree_y, num_nodes_y = EditDistanceKernel.nltk_to_zss(tree_y)

		normalized_score = simple_distance(tree_x, tree_y) / (num_nodes_x + num_nodes_y)

		return 1. - normalized_score # similarity

	