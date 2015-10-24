Warning: this code largely reuses the C-LDA code without refactoring, so it is even more unreadable and messier.

## Example usage

```
% ./stirling.py
```

To pre-compute the Stirling numbers, the numbers will not be computed in the inference procedure so please set the matrix dimension to some adequate value.

```
% python setup.py build_ext --inplace && ./example.py
% ./read.py > topics.html
```

`read.py` contains some code to print top words in html format to stdout.


## Parameters

**corpus_path**: points to the training file

**prefix**: prefix for the results

**test_path**: if not None, will calculate the held-out perplexity based on this
test file

**num_topics_0**: number of common topics

**num_topcis_c**: a list of number of non-common topics, same length as the number
of collections

**delta**: parameter of C-HDP, a list of two values

**beta**: parameter of C-HDP, single value or a list of values

**b**: parameter of C-HDP, a list of three values for [gamma, alpha_0, alpha_1]

**shape**: three float numbers for the gamma prior to update b

**scale**: float number for the gamma prior to update b

**n_iter**: number of iterations

**save_interval**: the interval to save intermediate results, not saving when set
to -1

**eval_interval**: evaluation interval, not evaluating when set to -1

**burn_in**: length of burn in period

