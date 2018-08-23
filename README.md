# Markov Recurrent Neural Network for language modeling
In this project, we implement Markov Recurrent Neural Network (MRNN) for language modeling with Penn Treebank.

Markov recurrent neural network (MRNN) explore the stochastic transitions in recurrent neural networks by incorporating the Markov property with discrete random variables. This model was proposed to deal with highly structured sequential data with complicated latent information. The discrete samples are drawn from the parameterized categorical distribution at each time step, and latent information is encoded by different state encoders depends on which state is selected.

<img src="Others/NTMCell.png" width="100%">



## Setting
- Hardware:
	- CPU: Intel Core i7-4930K @3.40 GHz
	- RAM: 64 GB DDR3-1600
	- GPU: GeForce GTX 1080ti
- Tensorflow 1.4.1
- Dataset
	- Penn Treebank

## Result
- 2D visualization of hidden state

|<img src="Others/tsne_LSTM.png" width="800">|
|:--------------------------------------------:|
|LSTM (K=1)|


<img src="Others/tsne_K=2.png" width="400">|<img src="Others/stat_K=2.png" width="400/">
:--------------------------------------------:|:----------------------------------------:
MRNN K=2|Statistic of states


<img src="Others/tsne_K=4.png" width="400">|<img src="Others/stat_K=4.png" width="400/">
:--------------------------------------------:|:----------------------------------------:
MRNN K=4|Statistic of states

<img src="Others/tsne_K=8.png" width="400">|<img src="Others/stat_K=8.png" width="400/">
:--------------------------------------------:|:----------------------------------------:
MRNN K=8|Statistic of states

<img src="Others/tsne_K=12.png" width="400">|<img src="Others/stat_K=12.png" width="400/">
:--------------------------------------------:|:----------------------------------------:
MRNN K=12|Statistic of states

<img src="Others/tsne_K=16.png" width="400">|<img src="Others/stat_K=16.png" width="400/">
:--------------------------------------------:|:----------------------------------------:
MRNN K=16|Statistic of states
