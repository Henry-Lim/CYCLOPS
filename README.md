## Cambridge UROP 2020: Application of CYCLOPS in identifying rhythms in biological data

_Author: Henry Lim_

### Background

Circadian rhythms influence many aspects of physiology and behavior, and modulate many processes in mammals including body temperature, blood pressure, and locomotor activity. Identifying molecular mechanisms in humans is challenging as existing large-scale datasets rarely include time of day. To address this problem, we combine understanding of periodic structure, evolutionary conservation, and unsupervised machine learning to order unordered human biopsy data along a periodic cycle. This project addresses the problem of inferring time labels from such data to identify circadian rhythms in genes of humans and other mammals.

The algorithm investigated in this project, cyclic ordering by periodic structure (CYCLOPS), utilises evolutionary conservation and machine learning to identify elliptical structure in high-dimensional data. From this structure, CYCLOPS estimates the phase of each sample. We validated CYCLOPS first using artificially-generated oscillatory data followed by temporally ordered mouse and human data and demonstrated its consistency.

### Introduction

The original authors of CYCLOPS implemented the autoencoder structure and conducted downstream analysis in Julia 0.3.10. The associated files are available for download on GitHub (https://github.com/ranafi/CYCLOPS). Much of the first 2-3 weeks was spent on understanding the code in Julia and writing a similar version of CYCLOPS in Python 3.6.10.

### Version Information

These are the versions of Python packages used in this project:

_Tensorflow_: 2.3.0 <br>
_Keras_: 2.4.3 <br>
_Scikit-learn_: 0.23.1 <br>
_Pandas_: 1.0.5 <br>
_Matplotlib_: 3.0.0 <br>
