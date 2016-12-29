For each query group Q and each group D_{i} in database.

Min:		 	

		$I_{Q} = argmin_{i}\{d(q_j,d_{ik})~|~ q_j \in Q, d_{ik} \in D_i, i \in [1, ..., N]\}$.

Vote: 				

		I_{Q} = Mode\{argmin_{i}\{d(q_j,d_{ik})~|~ d_{ik} \in D_i, i \in [1, ..., N]\}~|~q_j \in Q\}.

Vote+KNN:			

		I_{Q} = Mode\{argKmin_{i}\{d(q_j,d_{ik})~|~ d_{ik} \in D_i, i \in [1, ..., N]\}~|~q_j \in Q\},

		where argKmin are the arguments of K smallest values of the given set.

Mean:				

		I_{Q} = argmin_i \{ || \mu_Q - \mu_{D_i}||_2~|~i \in [1, …,  N]\}.

Hypercube:			

		I_{Q} = argmax_i \{\frac{Cube_Q ~\cap~ Cube_{D_i}}{Cube_{D_i}~\cap~ Cube_{D_i}} ~|~ i \in [1, …, N]\}, 

		where edge i of Cube_A is represented as \mu_{A_i} \pm K \times std(A_i).

KL-divegence:		

		I_{Q} = argmin_i\{D_{KL}(Q || D_i) ~|~ i \in [1, …, N]\}, 

		where D_{KL}(A||B) = \frac{1}{2}[tr(\Sigma_B^{-1}\Sigma_A) + (\mu_B-\mu_A)^T \Sigma_B^{-1} (\mu_B-\mu_A) - dim + ln(\frac{det(\Sigma_B)}{det(\Sigma_A)})].

Covariance:			

		I_{Q} = argmax_i \{\frac{tr(\Sigma_Q \times \Sigma_{D_i})}{||\Sigma_Q||_2 \times ||\Sigma_{D_i}||_2} ~|~ i \in [1, …, N]\}.

Standardized Mean:

		I_Q = argmin\{(\mu_Q - \mu_{D_i})^{T} S_i^{-1} (\mu_Q - \mu_{D_i}) ~|~ i \in [1, …, N]\},

		where S_i is a diagonal matrix and {S_i}_{jj} is \sigma_Q[j]^2 + \sigma_{D_i}[j]^2.

---

Standardized Mean + Normalization:

		I_Q = argmin\{ \frac{(\mu_Q - \mu_{D_i})^{T} S_i^{-1} (\mu_Q - \mu_{D_i})}{\sqrt{tr(R_i^2)}} ~|~ i \in [1, …, N]\},

		where S_i is the same matrix defined above and R_i = S_i^{-\frac{1}{2}} (\Sigma_Q + \Sigma_{D_i}) S_i^{-\frac{1}{2}}.

Normalized Mean:	

		I_{Q} = argmin_i\{\frac{||\mu_Q - \mu_{D_i}||_2^2}{\sqrt{tr(\Sigma_Q^2) + tr(\Sigma_{D_i}^2) + 2tr(\Sigma_Q \times \Sigma_{D_i})}} ~|~ i \in [1,…,N]\}, or

		I_{Q} = argmin_i\{\frac{||\mu_Q - \mu_{D_i}||_2^2}{\sqrt{\frac{tr(\Sigma_Q^2)}{N_Q^2} + \frac{tr(\Sigma_{D_i}^2)}{N_{D_i}^2} + 2\frac{tr(\Sigma_Q \times \Sigma_{D_i})}{N_QN_{D_i}}  } } ~|~ i \in [1,…,N]\}.

		If Q \sim N(\mu_Q, \Sigma_Q) and D_i \sim N(\mu_{D_i}, \Sigma_{D_i}), then  Q-D_i \sim N(\mu_Q - \mu_{D_i}, \Sigma_Q+\Sigma_{D_i}).




























# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
