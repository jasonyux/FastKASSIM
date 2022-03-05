# Usage

For the **first time**, please run the `download()` method that downloads the stanford parser and extracts it to be visible to CASSIM
```python
>>> import fcassim.FastCassim as fcassim
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
	>>> import fcassim.FastCassim as fcassim
	>>> fastcassim = fcassim.FastCassim(fcassim.FastCassim.NEW_FTK)
	>>> fastcassim.compute_similarity("Winter is leaving.", "Spring is coming.")
	1.0
	```
	(which defaults to use the parameters specified in the custome example below)
- custom configuration:
	```python
	>>> metric = fcassim.FastCassim.NEW_FTK
	>>> param = {
	...     "sigma": 1,
	...     "lmbda": 0.4,
	...     "use_new_delta": True
	... }
	>>> fastcassim = fcassim.FastCassim(metric)
	>>> fastcassim.set_params(**param)
	>>> fastcassim.compute_similarity("Winter is leaving.", "Spring is coming.")
	1.0
	```