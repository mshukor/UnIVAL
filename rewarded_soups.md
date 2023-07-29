# Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards

Here we provide the scripts to reproduce the results for the [Rewarded soups](https://github.com/alexrame/rewardedsoups) paper on Visual Grounding. Specifically, the model is optimized on different objectives; small, medium and large objects, and then all these models are averaged to get the best of all, without inference overhead. 

The main objective is to adapt the model accorind to user preferences, by choosing the most suitable averaging coefficients.

The training is done on the RefCOCO+ dataset.

The scripts to launch the training can be found in `run_scripts/refcoco/scst/`.
For example, to optimize for small objects:

```
sh unival_refcocoplus_acc0_5small_lreinf5.sh
```

To average different models after training, you can use the scripts in `preprocess/average_save_models`


