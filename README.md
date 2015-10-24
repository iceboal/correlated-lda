The implementation of “[**Fast, Flexible Models for Discovering Topic Correlation
across Weakly Related Collections**](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP179.pdf)” (EMNLP 2015).

Disclaimer: this research code is nasty, lack of proper design or comments. Use at your own risk :-)

## Prerequisites

Cython, CythonGSL

## Usage

Please refer to the README in each model folder for usage.

## Data format

In a corpus file, each line represents a document, which has words follow the
collection_id, all separated by space.

    <collection_id> <word_id> <word_id> <word_id>...

All ids begin with 0.

### Vocabulary file

The vocabulary file is used by read.py, each line is a word:

    <word0>
    <word1>
    ...

So `<word0>` has id = 0.

-----

If you use C-LDA/C-HDP for research purpose, please use the following citation:

    @InProceedings{zhang-EtAl:2015:EMNLP2,
      author    = {Zhang, Jingwei  and  Gerow, Aaron  and  Altosaar, Jaan  and  Evans, James  and  Jean So, Richard},
      title     = {Fast, Flexible Models for Discovering Topic Correlation across Weakly-Related Collections},
      booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
      month     = {September},
      year      = {2015},
      address   = {Lisbon, Portugal},
      publisher = {Association for Computational Linguistics},
      pages     = {1554--1564},
      url       = {http://aclweb.org/anthology/D15-1179}
    }

