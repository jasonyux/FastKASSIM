import nltk

from ..representation.treenode import TreeNode

class DeltaVector():
	NO_RESPONSE = -1
	def __init__(self) -> None:
		self.__map = dict()
		pass

	def get(self, i, j):
		key = tuple(sorted((str(i),str(j))))
		v = self.__map.get(key)
		return v or DeltaVector.NO_RESPONSE

	def add(self, i, j, v):
		key = tuple(sorted((str(i),str(j))))
		self.__map[key] = v
		return

	def print(self):
		print(self.__map)
		return

def to_treenode_tree(tree:nltk.tree, start_id:int=0):
	"""[summary]

	Args:
		tree (nltk.tree): root of a nltk.tree.Tree
		start_id (int, optional): start_id+1 will the smallest ID]. Defaults to 0

	Returns:
		(TreeNode, end_id): root of the tree in a TreeNode, largest id assigned
		Converts a tree of nltk.tree.Tree to a tree of TreeNodes. 
		The ID of each node will be assigned incrementally starting from start_id+1 in a DFS manner.
		Leave nodes/textual symbols are also wrapped as a TreeNode.
	"""
	if isinstance(tree, str):
		return TreeNode(start_id+1, tree, []), start_id+1
	child_nodes = []
	for child in tree:
		node, start_id = to_treenode_tree(child, start_id)
		child_nodes.append(node)
	# construct the tree node with its children correctly
	node = TreeNode(start_id+1, tree, child_nodes)
	return node, start_id+1

def find_all_node_pairs(tree_x:nltk.tree.Tree, tree_y:nltk.tree.Tree, include_leaves:bool):
	tree_x_root, end_id  = to_treenode_tree(tree_x, 0)
	node_x_list = tree_x_root.get_all_nodes(include_leaves)
	if tree_y != tree_x:
		tree_y_root, end_id  = to_treenode_tree(tree_y, end_id)
		node_y_list = tree_y_root.get_all_nodes(include_leaves)
	else:
		node_y_list = node_x_list
	ret = []
	for node_x in node_x_list:
		for node_y in node_y_list:
			ret.append((node_x, node_y))
	return ret

def find_common_nodes_by_production(tree_x:nltk.tree.Tree, tree_y:nltk.tree.Tree, include_leaves:bool) -> "list[tuple]":
	"""Table 1 Algorithm

	Returns:
		list: list of pairs of TreeNodes
	"""

	start_id = 0
	tree_x_root, end_id  = to_treenode_tree(tree_x, start_id)

	if tree_y != tree_x:
		tree_y_root, end_id  = to_treenode_tree(tree_y, end_id)
	else:
		tree_y_root = tree_x_root
	
	if include_leaves:
		nodes_x = tree_x_root.get_ordered_node_set_by_production() # figure out if those can be called or if need to implement
		nodes_y = tree_y_root.get_ordered_node_set_by_production()
	else:
		nodes_x = tree_x_root.get_ordered_node_set_by_production_ignoring_leaves()
		nodes_y = tree_y_root.get_ordered_node_set_by_production_ignoring_leaves()
		
	result = []
	index_x = 0
	index_y = 0
	
	while index_x < len(nodes_x) and index_y < len(nodes_y):
		if nodes_x[index_x][1] > nodes_y[index_y][1]:
			index_y += 1
		elif nodes_x[index_x][1] < nodes_y[index_y][1]:
			index_x += 1
		else:
			prev_index_y = index_y
			while index_x < len(nodes_x) and index_y < len(nodes_y) and nodes_x[index_x][1] == nodes_y[index_y][1]: # instead of equal want to compare productions & purge 
				while index_x < len(nodes_x) and index_y < len(nodes_y) and nodes_x[index_x][1] == nodes_y[index_y][1]:
					result.append((nodes_x[index_x][0], nodes_y[index_y][0]))
					index_y += 1
				index_x += 1
				index_y = prev_index_y
	return result