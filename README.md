# text-generation
By [Marco Boucas](http://marco.boucas.fr)

All the different analysis of the ability of NLP models to generate text (human readable)
We will go through different approaches (similar to the historic evolution of NLP).


## Table of Content

1. [Ngram](#ngram)
1. CNN
1. RNN (LSTM / GRU
1. Transformer
1. Specific papers

[**Vocabulary**](#vocabulary)

## Ngram

Ngrams are a group of n tokens together. For instance, in the sentence "Bob is in the kitchen and eats an apple", we have the following ngrams:

* 1-gram :
  * (Bob), (is), (in), (the), (kitchen), ...
* 2-gram
  * (Bob is), (is in), (in the), (the kitchen), ...
* 3-gram
  * (Bob is in), (is in the), (in the kitchen), ...

As you can see, increasing the value of n increase the number of possible groups, we will need to keep that in mind to avoid having million of groups ;)

