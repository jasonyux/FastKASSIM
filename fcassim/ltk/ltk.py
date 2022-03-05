from .representation.treenode import TreeNode
from .utils.tk_utils import DeltaVector, find_all_node_pairs, find_common_nodes_by_production
from nltk.tree import Tree

import math

class LabelTreeKernel():
	include_leaves = False # KELP has this default to True

	def __init__(self) -> None:
		pass

	@staticmethod
	def kernel(tree_x:Tree, tree_y:Tree, sigma:int, lmbda:float, use_new_delta=True) -> float:
		"""
        returns a normalized kernel
        """
		norm_factor = 1/math.sqrt(LabelTreeKernel.kernel_computation(tree_x, tree_x, sigma, lmbda, use_new_delta) \
			* LabelTreeKernel.kernel_computation(tree_y, tree_y, sigma, lmbda, use_new_delta))
		kern = LabelTreeKernel.kernel_computation(tree_x, tree_y, sigma, lmbda, use_new_delta)
		return norm_factor * kern
	
	@staticmethod
	def kernel_computation(tree_x:Tree, tree_y:Tree, sigma:int, lmbda:float, use_new_delta=True) -> float:
		delta_sum = 0.0
		delta_vector = DeltaVector()

		# configures which functions to use, either the original/Java impl or the new ones
		delta = LabelTreeKernel.delta_function if use_new_delta else LabelTreeKernel.delta_function_original
		find_common_nodes = find_all_node_pairs if use_new_delta else find_common_nodes_by_production

		node_pairs = find_common_nodes(tree_x, tree_y, LabelTreeKernel.include_leaves)
		for node_x, node_y in node_pairs:
			delta_sum += delta(node_x, node_y, sigma, lmbda, delta_vector)
		return delta_sum
	
	@staticmethod
	def delta_function(tree_x:TreeNode, tree_y:TreeNode, sigma:int, lmbda:float, delta_vector:DeltaVector) -> float:
		cached_value = delta_vector.get(tree_x.get_id(), tree_y.get_id())
		if (cached_value != delta_vector.NO_RESPONSE):
			return cached_value

		# if the node itself is different then GG!
		if tree_x.label() != tree_y.label():
			return 0

		# if include_leaves=True, then pre-terminals will NOT terminate and leave nodes will move on
		# if include_leaves=False, then pre-terminals will terminate here and leaves not even touched
		if LabelTreeKernel.include_leaves:
			if (not tree_x.has_children() and not tree_y.has_children()):
				ret = lmbda if tree_x.label() == tree_y.label() else 0
				delta_vector.add(tree_x.get_id(), tree_y.get_id(), ret)
				return ret
		else:
			if (tree_x.is_preterminal() and tree_y.is_preterminal()):
				ret = lmbda if tree_x.label() == tree_y.label() else 0
				delta_vector.add(tree_x.get_id(), tree_y.get_id(), ret)
				return ret

		prod = 1
		tree_x_children = tree_x.get_children()
		tree_y_children = tree_y.get_children()
		for child_x in tree_x_children:
			# to compute the number of SST/ST rooted at child_x, check with all of tree_y_children
			tmp = 0
			#print(f'start tmp={tmp} at {child_x}')
			for child_y in tree_y_children:
				#print(f'at tmp={tmp} at {child_y}')
				tmp += LabelTreeKernel.delta_function(child_x, child_y, sigma, lmbda, delta_vector)
			#print(f'ret tmp={tmp} at {child_x}')
			prod *= (sigma + tmp)
		
		delta_vector.add(tree_x.get_id(), tree_y.get_id(), lmbda * prod)
		return lmbda * prod

	@staticmethod
	def delta_function_original(tree_x:TreeNode, tree_y:TreeNode, sigma:int, lmbda:float, delta_vector:DeltaVector) -> float:
		cached_value = delta_vector.get(tree_x.get_id(), tree_y.get_id())
		if (cached_value != delta_vector.NO_RESPONSE):
			return cached_value
		
		if (tree_x.is_preterminal() and tree_y.is_preterminal()):
			delta_vector.add(tree_x.get_id(), tree_y.get_id(), lmbda)
			return lmbda
		
		prod = 1
		tree_x_children = tree_x.get_children()
		tree_y_children = tree_y.get_children()
		for i in range(len(tree_x_children)):
			x_child = tree_x_children[i]
			y_child = tree_y_children[i]
			#print(f"At: x={x_child}, y={y_child}; prod={prod}")
			#if (x_child.has_children() and y_child.has_children() and x_child.productions() == y_child.productions()):
			if (x_child.productions() == y_child.productions()):
				prod *= (sigma + LabelTreeKernel.delta_function(x_child, y_child, sigma, lmbda, delta_vector))
			else:
				#if (x_child.has_children() != y_child.has_children()):
				prod *= sigma
			#print(f"Ret: x={x_child}, y={y_child}; prod={prod}")
		
		delta_vector.add(tree_x.get_id(), tree_y.get_id(), lmbda * prod)
		return lmbda * prod

		
		