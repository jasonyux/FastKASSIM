# FastKASSIM: A Fast Kernel-Based Syntactic Similarity Metric

FastKASSIM [[1]](#1) - a fast, extensible metric for document-level syntactic similarity inspired by the Fast Tree Kernel [[2]](#2)  and [CASSIM](https://github.com/USC-CSSL/CASSIM) [[3]](#3)

# Table of Contents

- [FastKASSIM](#fastkassim)
- [Usage](#usage)
- [Using your own Tree Kernel](#using-your-own-tree-kernel)
  * [Adding Tree Kernels to FastKassim](#adding-tree-kernels-to-fastkassim)
- [Paper/References](#references)
# Usage

For the **first time**, please run the `download()` method that downloads and extracts the Stanford Parser
```python
>>> import fkassim.FastKassim as fkassim
>>> fkassim.download()
Downloading https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip
Extracting
Cleaning up
Done
```
note that since `https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip` is large in size, it might take around a minute to download.

Then, example usages would be:
- **quickstart**:
	```python
	>>> import fkassim.FastKassim as fkassim
	>>> FastKassim = fkassim.FastKassim(fkassim.FastKassim.LTK)
	>>> FastKassim.compute_similarity("Winter is leaving.", "Spring is coming.")
	1.0
	```
	(which defaults to use the parameters specified in the custome example below)

- **custom configuration**:
	```python
	>>> import fkassim.FastKassim as fkassim
	>>> metric = fkassim.FastKassim.LTK
	>>> param = {
	...     "sigma": 1,
	...     "lmbda": 0.4,
	...     "average": False
	... }
	>>> FastKassim = fkassim.FastKassim(metric)
	>>> FastKassim.set_params(**param)
	>>> FastKassim.compute_similarity("Winter is leaving.", "Spring is coming.")
	1.0
	```
	currently implemented tree similarity metrics include:
	- `FastKassim.LTK`: label-based tree kernel [[1]](#1)
	- `FastKassim.FTK`: fast tree kernel [[2]](#2)
	- `FastKassim.ED`: normalized edit distance [[3]](#3)

- **Need to recompute lots of parse trees (e.g., pairwise comparisons)? Try using customizable document parsing**:
	```python
	>>> import fkassim.FastKassim as fkassim
	>>> FastKassim = fkassim.FastKassim(fkassim.FastKassim.LTK)
	>>> doc1 = """
	... Harpers, Harpers, they really care. Harpers, Harpers, stay in motion.
	... """
	>>> doc2 = """
	... Harpers, Harpers, they really heal. Harpers, Harpers, stay in movement.
	... """
	>>> # parse document into list of parse trees
	>>> # defaults to use FastKassim's parser and tokenizer
	>>> doc1_parsed = FastKassim.parse_document(doc1)
	>>> doc2_parsed = FastKassim.parse_document(doc2)
	>>> 
	>>> # compute similarity score directly from the parsed trees
	>>> # notice that this step will be very fast due to the usage of kernels!
	>>> FastKassim.compute_similarity_preparsed(doc1_parsed, doc2_parsed)
	0.7717689799810963
	```
	here `parse_document(doc, tokenizer=None, parser=None)` offers the advantage of:
	- being able to customize what parser and sentence tokenizer you want to use
	- since parsing long documents may take a long time, this allows you to multithread/save parses for future use
	
	Resultwise, the above will be the same as doing:
	```python
	>>> import fkassim.FastKassim as fkassim
	>>> FastKassim = fkassim.FastKassim(fkassim.FastKassim.LTK)
	>>> doc1 = """
	... Harpers, Harpers, they really care. Harpers, Harpers, stay in motion.
	... """
	>>> doc2 = """
	... Harpers, Harpers, they really heal. Harpers, Harpers, stay in movement.
	... """
	>>> FastKassim.compute_similarity(doc1, doc2)
	0.7717689799810963
	```

# Using your own Tree Kernel
Another goal of this project is to allow for any similarity metrics (between two parse trees) to be used. You can implement and use your own metric/kernel function by:
1. implement a kernel class, that has:
	- a kernel method that computes the normalized similarity score between two parse trees `kernel(tree_x:Tree, tree_y:Tree, **params)`
2. construct a `Kassim` object and pass in your kernel instance and kernel related `params`

*For Example*
1. Here a simple class that outputs the sum of `value1` and `value2` as similarity score
	```python
	import nltk

	class DummyKernel(object):
		NAME = "DummyKernel"

		def __init__(self):
			pass

		def kernel(self, tree_x:nltk.Tree, tree_y:nltk.Tree, **params):
			"""
			returns a normalized edit distance score
			"""
			return params["value1"] + params["value2"]
	```
2. Then, what `params` will be passed into the `kernel` function is specified during the intialization of the `MyKassim` class:
	```python
	from fkassim.kassim.Kassim import Kassim

	MyKassim = Kassim(Kernel=DummyKernel(), value1=0.5, value2=0.4)
	```
	so that all later computations will only involve using `MyKassim`:
	```python
	doc1 = "Winter is leaving."
	doc2 = "Spring is coming."

	MyKassim.compute_similarity(doc1, doc2)
	# returns 0.9
	```
	Both `parse_document` and `compute_similarity_preparsed` will also be available:
	```python
	doc1 = """
		Harpers, Harpers, they really care. Harpers, Harpers, stay in motion.
	"""
	doc2 = """
		Harpers, Harpers, they really heal. Harpers, Harpers, stay in movement.
	"""
	# parse document into list of parse trees
	doc1_parsed = MyKassim.parse_document(doc1)
	doc2_parsed = MyKassim.parse_document(doc2)

	MyKassim.compute_similarity_preparsed(doc1_parsed, doc2_parsed)
	# returns 0.9
	```

## Adding Tree Kernels to FastKassim
Currently we have three metrics implemented for measuring syntax similarity between two documents, and they all come down to measuring syntax similarity between the parse trees of two sentences.

Therefore, if you want to invent your own metric, you can do so by:
1. implement a class that contains:
	- static variables representing its name and ID
	- a **parameter configuration method** with the signature `config(metric, **user_configs)`
	- a **kernel method** that computes the normalized similarity score between two parse trees `kernel(tree_x:Tree, tree_y:Tree, **params)`
2. register your class by updating the `fkassim/FastKassim.py` file:
	- add your class's ID to the static variables inside the `Fastkassim` class:
		```python
		class FastKassim(kassim):
			LTK = LabelTreeKernel.LTK
			FTK = LabelTreeKernel.FTK
			ED = EditDistanceKernel.EDK
			# new ones here
		```
	- update the `__configure` and `__configure_kernel` method call

*For example:* implementing the normalized edit distance kernel `EditDistanceKernel`

1. the class will be created under `fkassim/edk/edk.py`, and the key components will be:
	```python
	class EditDistanceKernel(object):
		NAME = "EditDistanceKernel" # your kernel's name
		EDK = 3 # a unique ID

		def __init__(self) -> None:
			pass

		@staticmethod
		def config(metric, **user_configs):
			"""returns customized parameters but relevant to the current kernel

			Returns:
				dict: parameters relevant to EditDistanceKernel
			"""
			default_params = {
				"average": False # not really used, here for demonstration purposes
			}
			# filter accepted params
			# eventually this will be passed into kernel(tree_x, tree_y, **params) call 
			# as the params field
			conf_params = {}
			for k,v in default_params.items():
				if user_configs.get(k) is None:
					conf_params[k] = v
				else:
					conf_params[k] = user_configs[k]
			
			return conf_params

		@staticmethod
		def kernel(tree_x:Tree, tree_y:Tree, **params) -> float:
			"""
			returns a normalized edit distance score
			"""
			tree_x, num_nodes_x = EditDistanceKernel.nltk_to_zss(tree_x)
			tree_y, num_nodes_y = EditDistanceKernel.nltk_to_zss(tree_y)

			normalized_score = simple_distance(tree_x, tree_y) / (num_nodes_x + num_nodes_y)

			return 1. - normalized_score # similarity between 0 and 1
	```
	(certain helper methods are omitted, only required methods/variables are shown here)

2. Then, register this new class into `fkassim/FastKassim.py` by:

	(only necessary changes are shown here)
	```python
	class FastKassim(kassim):
		LTK = LabelTreeKernel.LTK
		FTK = LabelTreeKernel.FTK
		ED = EditDistanceKernel.EDK # 1. ADD your class's ID here

		def __configure_kernel(self, metric, **params):
			if self.__metric == FastKassim.LTK:
				self.__kernel = LabelTreeKernel
				return LabelTreeKernel.config(metric, **params)
			elif self.__metric == FastKassim.FTK:
				self.__kernel = LabelTreeKernel
				return LabelTreeKernel.config(metric, **params)
			elif self.__metric == FastKassim.ED: # 2. ADD your class and config method here
				self.__kernel = EditDistanceKernel
				return EditDistanceKernel.config(metric, **params)
			return

		def __configure(self, metric, params):
			"""
			Main entry point for configuring kernel method, parameters, and etc
			"""
			# 3. Update the error condition
			if metric < FastKassim.LTK or metric > FastKassim.ED: 
				raise Exception(f"""
				Please specify metric to be:
					FastKassim.LTK or {FastKassim.LTK};
					FastKassim.FTK or {FastKassim.FTK};
					FastKassim.ED or {FastKassim.ED};
				""")
			
			self.__metric = metric
			conf_params = self.__configure_kernel(metric, **params)
			return metric, conf_params
	```

> Note: if you find something isn't quite right, turn on `logging.basicConfig(level=logging.INFO)` to get messages printed out about which kernel is configured and what parameters are used.

# References
If FastKASSIM is useful in any work resulting in publication, please cite the FastKASSIM, CASSIM and FTK papers. FastKASSIM would not exist without CASSIM and FTK.

<a id="1">[1]</a> 
FastKASSIM: A Fast Tree Kernel-Based Syntactic Similarity Metric<br>
Maximillian Chen*, Caitlyn Chen*, Xiao Yu*, Zhou Yu<br>
arXiv preprint (2022)<br>
https://arxiv.org/abs/2203.08299
```tex
@article{chen2022fastkassim,
  title={FastKASSIM: A Fast Tree Kernel-Based Syntactic Similarity Metric},
  author={Chen, Maximillian and Chen, Caitlyn and Yu, Xiao and Yu, Zhou},
  journal={arXiv preprint arXiv:2203.08299},
  year={2022}
}
```

<a id="2">[2]</a> 
Making Tree Kernels Practical for Natural Language Learning<br>
Alessandro Moschitti<br>
Proceedings of the 11th Conference of the European Chapter of the Association for Computational Linguistics (2006)<br>
https://aclanthology.org/E06-1015.pdf
```tex
@inproceedings{moschitti2006making,
  title={Making tree kernels practical for natural language learning},
  author={Moschitti, Alessandro},
  booktitle={11th conference of the European Chapter of the Association for Computational Linguistics},
  pages={113--120},
  year={2006}
}
```

<a id="3">[3]</a> 
Conversation level syntax similarity metric<br>
Reihane Boghrati, Joe Hoover, Kate M. Johnson, Justin Garten, and Morteza Dehghani<br>
Behavior Research Methods 50, 1055–1073 (2018)<br>
https://doi.org/10.3758/s13428-017-0926-2
```tex
@article{boghrati2018conversation,
  title={Conversation level syntax similarity metric},
  author={Boghrati, Reihane and Hoover, Joe and Johnson, Kate M and Garten, Justin and Dehghani, Morteza},
  journal={Behavior research methods},
  volume={50},
  number={3},
  pages={1055--1073},
  year={2018},
  publisher={Springer}
}
```
