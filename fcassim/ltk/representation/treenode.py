from nltk.tree import Tree
from nltk.grammar import Production

class TreeNode():
	def __init__(self, id:int, tree:Tree, children=None) -> None:
		self.__id = id
		self.__tree = tree
		self.__children = children
		self.__productions = self.__get_productions()
		return
	
	def __get_productions(self):
		ret = []
		tree = self.__tree
		if isinstance(tree, str):
			return [tree]
		productions = tree.productions()
		# check and remove textual leaf nodes in production
		for prod in productions:
			if prod.is_lexical():
				prod = Production(prod.lhs(), [])
			ret.append(prod)
		return ret

	def get_id(self):
		return self.__id
		#return self.__tree

	def get_children(self):
		return self.__children
	
	def is_preterminal(self):
		return not self.has_children() or self.__tree.height() == 2

	def has_children(self):
		return len(self.__children) > 0

	def productions(self):
		return self.__productions

	def get_all_nodes(self, include_leaves=False):
		ret = TreeNode.__bfs_get_all_nodes(self, [], include_leaves)
		ret.insert(0, self)
		return ret

	@staticmethod
	def __bfs_get_all_nodes(root, res_buffer, include_leaves):
		queue = []
		queue.append(root)
		# init queue
		while len(queue) != 0:
			curr = queue.pop(0)
			for child in curr.get_children():
				# skip if NOT include leaves AND is a leaf
				if not include_leaves and not child.has_children():
					continue
				queue.append(child)
				res_buffer.append(child)
		return res_buffer

	def label(self):
		# is string is has no children
		if not self.has_children():
			return self.__tree
		return self.__tree.label()
	
	def get_ordered_node_set_by_production(self) -> "list[TreeNode]": 
        # returns the complete set of nodes ordered alphabetically by production string ignoring content of leaves (need to purge here)
        
        # can figure out that a node is not a leaf if the height of the node is 1

		def get_ordered_node_set_by_production_helper(subtree: TreeNode, node_prods: list):
            # do the purging here
			for child in subtree.get_children(): # covers checking for non-terminals, up until preterminal
				node_prods.append((child, [str(rule) for rule in child.productions()]))
				get_ordered_node_set_by_production_helper(child, node_prods)
		
		node_prods = []
		node_prods.append((self, [str(rule) for rule in self.productions()]))
		
		get_ordered_node_set_by_production_helper(self, node_prods)
		node_prods.sort(key=lambda x:x[1])
		return node_prods

	def get_ordered_node_set_by_production_ignoring_leaves(self) -> "list[TreeNode]":
        # returns the complete set of nodes ordered alphabetically by production string
		
		node_prods = []
		node_prods.append((self, [str(rule) for rule in self.productions()]))
		
		def get_ordered_node_set_by_production_ignoring_leaves_helper(subtree: Tree, node_prods: list):
			
			for child in subtree.get_children():
				if child.has_children(): # covers checking for non-terminals, up until preterminal
					node_prods.append((child, [str(rule) for rule in child.productions()]))
					get_ordered_node_set_by_production_ignoring_leaves_helper(child, node_prods)

		get_ordered_node_set_by_production_ignoring_leaves_helper(self, node_prods)		
		
        # sort list of tuples based on child.productions() elements
		node_prods.sort(key=lambda x:x[1])
		return node_prods

	def __str__(self):
		return self.__tree.__str__()


        