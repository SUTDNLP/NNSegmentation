NNSegmentation
======
NNSegmentation is a package for Word Segmentation using neural networks based on package [LibN3L](https://github.com/SUTDNLP/LibN3L). 
It includes different combination of ***Neural network architectures*** (TNN, RNN, GatedNN, LSTM and GRNN) with ***Objective function***(Softmax, CRF Max-Margin, CRF Maximum Likelihood).
It also provides the capability of combination of ***Sparse feature*** along with above models. 
In addition, this package can easily support various user-defined neural network structures.

Performance
======
Please read Table 4 in [LibN3L: A lightweight Package for Neural NLP](https://github.com/SUTDNLP/LibN3L/blob/master/description\(expect%20for%20lrec2016\).pdf).

Compile
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and compile it. 
* Open [CMakeLists.txt](CMakeLists.txt) and change "../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.
* 
    cmake .

*
    make

Example
======
This example shows how to train three Chinese word segmentation models for the pku corpus of the Sighan Bakeoff 2005 dataset.  
These models are
* SparseCRFMMLabler which only considers the sparse features and works like a CRF model
* LSTMCRFMMLabeler which only uses neural embeddings as input and employs CRF Maximum Likelihood as training objective.  
* SparseLSTMCRFMMLabeler which supports both neural embeddings and sparse features and also employs CRF Maximum Likelihood as training objective.  

This example data contains  
* Sparse Features ["train.feats"](example/pku/pku.sample.train.feats), ["dev.feats"](example/pku/pku.sample.dev.feats) and ["test.feats"](example/pku/pku.test.feats). The training features and dev features are extracted only from a subset of the pku corpus.   
* Character Unigram Embedding ["char.vec"](example/embeddings/char.vec)
* Character Bigram Embedding ["bi.vec"](example/embeddings/bichar.vec)
* Character Trigram Embedding ["tri.vec"](example/embeddings/trichar.vec)
* Parameter Setting File ["sparse"](example/options/option.sparse) for SparseCRFMMLabler, ["lstm"](example/options/option.sparse) for LSTMCRFMMLabeler and ["sparselstm"](example/options/option.sparse+lstm) for SparseLSTMCRFMMLabler.

For more details about the example, please read the example ["readme"](example/readme.md).
