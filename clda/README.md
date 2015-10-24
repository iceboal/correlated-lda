## Example usage

```
% python setup.py build_ext --inplace && ./example.py
% ./read.py > topics.html
```

`read.py` contains some code to print top words in html format to stdout.


## Parameters

**corpus_path**: points to the training file

**prefix**: prefix for the results

**test_path**: if not None, will calculate the held-out perplexity based on this test file

**num_topics_0**: number of common topics

**num_topcis_c**: a list of number of non-common topics, same length as the number of collections

**alpha**: not important since the model estimates it

**delta**: parameter of C-LDA, a list of two values

**beta**: parameter of C-LDA, single value or a list of values

**n_worker**: number of threads

**n_iter**: number of iterations

**save_interval**: the interval to save intermediate results, not saving when set to -1

**eval_interval***: evaluation interval, not evaluating when set to -1

**burn_in**: length of burn-in period

