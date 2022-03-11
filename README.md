# FastKassim

FastKassim - a fast metric for document-level syntactic similarity inspired by the Fast Tree Kernel [[1]](#1)  and [CASSIM](https://github.com/USC-CSSL/CASSIM) [[2]](#2)

# Usage

For the **first time**, please run the `download()` method that downloads the stanford parser and extracts it to be visible to CASSIM
```python
>>> import fcassim.FastKassim as fcassim
>>> fcassim.download()
Downloading https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip
Extracting
Cleaning up
Done
```
note that since `https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip` is large in size, it might take around a minute to download.

Then, example usages would be:
- **quickstart**:
	```python
	>>> import fcassim.FastKassim as fcassim
	>>> FastKassim = fcassim.FastKassim(fcassim.FastKassim.LTK)
	>>> FastKassim.compute_similarity("Winter is leaving.", "Spring is coming.")
	1.0
	```
	(which defaults to use the parameters specified in the custome example below)
	
- **Need to recompute lots of parse trees (e.g., pairwise comparisons)? Try using customizable document parsing**:
	```python
	>>> import fcassim.FastKassim as fcassim
	>>> FastKassim = fcassim.FastKassim(fcassim.FastKassim.LTK)
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
	>>> import fcassim.FastKassim as fcassim
	>>> FastKassim = fcassim.FastKassim(fcassim.FastKassim.LTK)
	>>> doc1 = """
	... Harpers, Harpers, they really care. Harpers, Harpers, stay in motion.
	... """
	>>> doc2 = """
	... Harpers, Harpers, they really heal. Harpers, Harpers, stay in movement.
	... """
	>>> FastKassim.compute_similarity(doc1, doc2)
	0.7717689799810963
	```
	
- **custom configuration**:
	```python
	>>> import fcassim.FastKassim as fcassim
	>>> metric = fcassim.FastKassim.LTK
	>>> param = {
	...     "sigma": 1,
	...     "lmbda": 0.4,
	...     "average": False
	... }
	>>> FastKassim = fcassim.FastKassim(metric)
	>>> FastKassim.set_params(**param)
	>>> FastKassim.compute_similarity("Winter is leaving.", "Spring is coming.")
	1.0
	```

# References
If FastKASSIM is useful in any work resulting in publication, please cite the FastKASSIM, CASSIM and FTK papers. FastKASSIM would not exist without CASSIM and FTK.

<a id="1">[1]</a> 
Moschitti, Alessandro. (2006). Making Tree Kernels Practical for Natural Language Learning.. Proceedings of the 11th Conference of the European Chapter of the Association for Computational Linguistics. https://aclanthology.org/E06-1015.pdf

<a id="2">[2]</a> 
Boghrati, R., Hoover, J., Johnson, K.M. et al. Conversation level syntax similarity metric. Behav Res 50, 1055–1073 (2018). https://doi.org/10.3758/s13428-017-0926-2
