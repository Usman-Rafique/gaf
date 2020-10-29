# Generative Appearance Flow: A Hybrid Approach for Outdoor View Synthesis
&emsp; &emsp; [Muhammad Usman Rafique](http://urafique.com), &emsp; [Hunter Blanton](https://hblanton.github.io/), &emsp; [Noah Snavely](http://www.cs.cornell.edu/~snavely/), &emsp; [Nathan Jacobs](https://jacobsn.github.io/)

##  &emsp; &emsp; [Paper](https://www.bmvc2020-conference.com/assets/papers/0055.pdf) &emsp;  [Supplemental](https://www.bmvc2020-conference.com/assets/supp/0055_supp.zip) &emsp;      [Project](http://urafique.com/gaf/) &emsp;  [Talk](https://www.bmvc2020-conference.com/conference/papers/paper_0055.html)


![result1](https://github.com/Usman-Rafique/Usman-Rafique.github.io/blob/master/images/animation_AF_Plus01.gif)
![result1](https://github.com/Usman-Rafique/Usman-Rafique.github.io/blob/master/images/animation_AF_Plus35.gif)


## Pre-trained Model
Pretrained models are included in this repo under `checkpoints/`

## Using the Code
This repository contains scripts for training, evaluation, and generating visual results. The settings are stored in the file `config.py`. Before running any training or evaluation, please make sure that settings in `config.py` are correct. 

### Models
This repository comes with these models from our [paper]((https://www.bmvc2020-conference.com/assets/papers/0055.pdf)):
- Improved appearance flow: AF++
- Flow-guided direct synthesis: FDS
- Generative appearance flow: GAF

### Training
For training, you can specify a directory through `cfg.train.out_dir` (in `config.py`). Trained models and training logs will be saved in this directory. There are three models, which are trained sequentially AF++, FDS, and GAF.
1. AF++: run `python3 train_AF_plus.py`. There is no prerequisite to train AF++.
2. FDS: run `python3 train_FDS.py`. This requires a trained AF++. We have included a trained AF++ in `checkpoints/`, this will be loaded by default.
3. GAF: run `python3 train_GAF.py`. Trained AF++ and FDS are required. These are included in `checkpoints/` and will be loaded by default.

### Evaluation & Visualization
For quantitative evaluation, or to generate visual results, please select the correct directory in which trained models are saved, or set `cfg.train.out_dir='checkpoints'`. Also, make sure to set the desired network through `cfg.model.name`.

For visual results, run `python3 visualize.py`. For quantitative evaluation `python3 eval_trained.py`.

## BPS Dataset
All panoramic images have been north-aligned, cropped, and resized to the size 960 (width) x 160 (height). There are a total of 44 092 examples in the dataset; every example has two images and the relative camera transformations. 

We have collected the dataset from google street view images. To ensure fair use and avoid duplication, please send an email to usman dot rafique @ uky . edu. We will share the dataset link by email. 

## Permission
The code is released only for academic and research purposes.

## Recommended citation
<pre>
@inproceedings{rafique2020gaf,
  title={Generative Appearance Flow: A Hybrid Approach for Outdoor View Synthesis},
  author={Rafique, M. Usman and Blanton, Hunter and Snavely, Noah and Jacobs, Nathan},
  booktitle={The British Machine Vision Conference},
  year={2020}
}
</pre>
