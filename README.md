# stacked_capsule_autoencode_pl
This code refers to the [official google-research repository](https://github.com/google-research/google-research/tree/master/stacked_capsule_autoencoders).

This is a pytorch_lightning (pytorch) implemention of the Stacked Capsule Autoencoder (SCAE), which was introduced in the following paper:[A. R. Kosiorek](http://akosiorek.github.io/), [Sara Sabour](https://ca.linkedin.com/in/sara-sabour-63019132), [Y. W. Teh](https://www.stats.ox.ac.uk/~teh/), and [Geoffrey E. Hinton](https://vectorinstitute.ai/team/geoffrey-hinton/), ["Stacked Capsule Autoencoders"](https://arxiv.org/abs/1906.06818).

* Note that this code still exist bugs, we will fix it in the future. 


# Our environment
 * 'torch'==1.16
 * 'pytorch_lightning'==0.9.0
 * 'tensorboard'==1.15.2
 * 'numpy'==1.17

# Understanding the Code
  * The training loop is defined in `main.py`.
  * The model is built in `model/SCAE.py`.
  * Using `tensorboard --logdir=./lightning_log` to monitor the procedure of training.
