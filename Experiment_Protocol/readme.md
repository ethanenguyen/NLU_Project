The papers : 
* http://people.tamu.edu/~xhuang/Xiao_WSDM19.pdf
* https://arxiv.org/pdf/2001.11985.pdf

1. KG Embedding Based Question Answering  (ZichaoHuang)
https://github.com/xhuang31/KEQA_WSDM19


2. The Python's TransE using Pytorch
https://github.com/xjdwrj/TransE-Pytorch

3. Fine tune BERT for QA with SQUAD
https://hackernoon.com/nlp-tutorial-creating-question-answering-system-using-bert-squad-on-colab-tpu-1utp3352

4. Fine tune BERT for NER
https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/

5. Trains Bert over Freebase knowledge graph
https://arxiv.org/pdf/1909.03193.pdf and the code
https://github.com/yao8839836/kg-bert

-----------------------------------------------------------------------------------------------------------

1. KG Embedding using TransE, TransH, TransR, and PTransE ( C++, not Python )
https://github.com/thunlp/KB2E

3. The Python's TransE from ZichaoHuang ( TensorFlow, not Pytorch )
https://github.com/ZichaoHuang/TransE

Notice: transE_emb.py
I didn't find the ‘ transE_emb.py ’ file in your code, so I would like to ask how the initialization vector of transE is represented in the training, or can you give me the ‘ transE_emb.py ’ file? At the same time, I have a question about KEQA. The vectorized representation obtained after KEQA is not in the same vector space as the embedded representation TransE, so when the Euclidean distance between the two is found, will there be an error match? **

@xhuang31

Owner
xhuang31 commented on Sep 22, 2019
Thank you so much for the comments.

Knowledge graph embedding is not the focus of our paper. For the TransE embedding, we directly use the implementation from https://github.com/ZichaoHuang/TransE. I cannot put "transE_emb.py" in this repo since the contribution should be credited to https://github.com/ZichaoHuang/TransE.

Yes, you are right. It is not an error match, but an inaccurate match. Our goal is to make the predicted entity representation as close as possible to the representation of the correct head entity. The goal is NOT to recover exactly the same representation, but a close one, so that the learned vector could be used as a pointer to lead us to the current head entity.


