# FastKassim

FastKassim - a fast metric for document-level syntactic similarity based on Tree kernel [[1]](#1)  and [CASSIM](https://github.com/USC-CSSL/CASSIM) [[2]](#2)

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
- quickstart:
	```python
	>>> import fcassim.FastKassim as fcassim
	>>> FastKassim = fcassim.FastKassim(fcassim.FastKassim.LTK)
	>>> FastKassim.compute_similarity("Winter is leaving.", "Spring is coming.")
	1.0
	```
	(which defaults to use the parameters specified in the custome example below)
- custom configuration:
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
<a id="1">[1]</a> 
Moschitti, Alessandro. (2006). Making Tree Kernels Practical for Natural Language Learning.. Proceedings of the 11th Conference of the European Chapter of the Association for Computational Linguistics. https://aclanthology.org/E06-1015.pdf

<a id="2">[2]</a> 
Boghrati, R., Hoover, J., Johnson, K.M. et al. Conversation level syntax similarity metric. Behav Res 50, 1055–1073 (2018). https://doi.org/10.3758/s13428-017-0926-2