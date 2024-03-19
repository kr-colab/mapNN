# mapNN
Neural network for estimating demographic maps from SNPs
- [mapNN](#mapnn)
  - [How to cite](#how-to-cite)
  - [Install instructions](#install-instructions)
  - [Usage](#usage)
    - [Creating training maps](#creating-training-maps)
    - [Simulation](#simulation)
    - [Preprocessing](#preprocessing)
    - [Training](#training)
    - [Prediction](#prediction)
    - [Empirical analysis](#empirical-analysis)
  - [References](#references)



## How to Cite
Estimation of spatial demographic maps from polymorphism data using a neural network

Chris C R Smith, Gilia C Patterson, Peter L. Ralph, Andrew D. Kern

bioRxiv 2024.03.15.585300; doi: https://doi.org/10.1101/2024.03.15.585300






## Install instructions

```bash
conda create -n mapnn python=3.9 --yes
conda activate mapnn
pip install --upgrade pip
pip install -r requirements/development.txt
```

Test command:

```bash
wget http://sesame.uoregon.edu/~chriscs/mapNN/Examples.zip
unzip Examples.zip 
python mapnn.py --train --out Examples/Train/ --seed 123 --num_snps 5000 --n 100
```

Additional installation instructions for GPUs:

```bash
mamba install cudatoolkit=11.8.0 cuda-nvcc -c conda-forge -c nvidia --yes
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.1
mkdir -p $CONDA_PREFIX/bin/nvvm/libdevice/
ln -s $CONDA_PREFIX/nvvm/libdevice/libdevice.10.bc $CONDA_PREFIX/bin/nvvm/libdevice/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/bin/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Test command:

```bash
python mapnn.py --train --out Examples/Train/ --seed 123 --num_snps 5000 --n 100 --gpu any
```

And simultaneously run `nvidia-smi` or `gpustat` to make sure the GPU is firing.











## Usage

The `mapNN` program is used for several steps in an analysis workflow:
(i) preprocessing training data, (ii) training the neural network, (iii) prediction/validation on simulated data, and (iv) empirical predictions.
See below sub-sections for example commands and explanations of the command line flags for each step.

To generate training data we use SLiM (Haller and Messer, 2023). Check out the below subsections to see how we generate input-maps for SLiM, and for an example command using the	SLiM recipe from Smith et al. (2024).

Before running the below commands, first create a testing directory and load up your conda env:

```bash
conda activate mapnn
mkdir tempout
```






### Creating training maps

usage:

```bash
python create_maps.py \
       --out <str> \
       --seed <int> \
       --w <int> \
       --min_c1 <float> \
       --max_c1 <float> \
       --min_c2 <float> \
       --max_c2 <float> \
       --png \
       --max_degree <int>
```

- `out`: output prefix
- `seed`: random number seed
- `w`: map width
- `min_c1`: minimum of the first input channel
- `max_c1`: max of the first input channel
- `min_c`: minimum of the first second channel
- `max_c`: max of the second input channel
- `png`: outputs a PNG rendering of the map (in addition to .npy array)
- `max_degree`: max degree of polynomial curve


The below command can be used to make a 50x50 map with two channels---e.g., one channel for dispersal, and one for carrying capacity---where values in the first channel vary between 0.73 and 3.08, and the second channel varies between 4 and 40.

example:

```bash
python create_maps.py \
       --out tempout/mapnn \
       --seed 123 \
       --w 50 \
       --min_c1 0.73 \
       --max_c1 3.08 \
       --min_c2 4 \
       --max_c2 40 \
       --png \
       --max_degree 3
```










### Simulation

Ideally, a custom simulation model should be used to produce training data, one that is tailored to your study system.
As a template, have a look at our SLiM recipes `benchmark.slim` and `wolves.slim`.

usage:
```bash
slim \
     -d "MAP_FILE_0='<str>'" \
     -d "MAP_FILE_1='<str>'" \
     -d "OUTNAME='<str>'" \
     -d SEED=<int> \
     -d SI=<float> \
     -d SM=<float> \
     -d maxgens=<int> \
     benchmark.slim
```

- `MAP_FILE_0`: path to dispersal csv
- `MAP_FILE_1`: path to density csv
- `OUTNAME`: path to output
- `SEED`: random number seed
- `SI`: interaction-sigma
- `SM`: mating-sigma
- `maxgens`: number of slim ticks before the end of simulation


First install SLiM:
```bash
mamba install slim==4.0.1 -c conda-forge
```

example:

```bash
slim \
     -d "MAP_FILE_0='tempout/mapnn_123_disp.csv'" \
     -d "MAP_FILE_1='tempout/mapnn_123_dens.csv'" \
     -d "OUTNAME='tempout/sim'" \
     -d SEED=123 \
     -d SI=1 \
     -d SM=1 \
     -d maxgens=10 \
     SLiM_recipes/benchmark.slim
```


Presuming that not all trees have coalesced we will need to recapitate the tree sequence.
For more information see the [msprime manual](https://tskit.dev/msprime/docs/stable/ancestry.html#continuing-forwards-time-simulations-recapitating) or [pyslim](https://tskit.dev/pyslim/docs/latest/tutorial.html#sec-tutorial-recapitation).

This can be done using a function in the `mapNN` repo:

```python
from process_input import recap
recap("tempout/sim_123.trees", "tempout/recap_123.trees", seed=123)
```











### Preprocessing

Assuming you are working with tree sequences (although, tree sequences are not necessary for `mapNN`), we will need to do additional preprocessing steps to format the input data that `mapNN` expects.
In particular, we need to sample individuals, simulate neutral mutations, and create three input files for each dataset: a genotype matrix, locations table, and the target map.

This can be done using the "preprocess" `mapNN` function:


usage:

```bash
python mapnn.py \
       --preprocess \
       --out <str> \
       --simid <int> \
       --seed <int> \
       --n <int> \
       --num_snps <int> \
       --tree_list <str> \
       --target_list <str> \
       --slim_width <float> \
       --map_width <float> \
       --sample_grid <int>
```

Since this is the first `mapNN` command we've introduced, there are several command line flags we need to describe:

- `preprocess`: tells `mapNN` sample and preprocess individuals from a tree sequence file.
- `out`: path to output directory (created during run).
- `simid`: simulation identifier (we usually set this to the random number seed used in the preceding steps, i.e., 123 in this case).
- `seed`: random number seed. Can be changed to take multiple samples from a tree sequence.
- `n`: number of individuals to sample.
- `num_snps`: number of	snps to	simulate.
- `tree_list`: path to a list of tree sequences (order should match the target_list)
- `target_list`: path to a list of target maps (order should match the tree_list)
- `slim_width`: the width of the habitat from SLiM. Used for re-scaling the locations (although, in this example slim_width was the same as the width of the target map)
- `map_width`: width of the target map
- `sample_grid`: coarseness of the spatial sampling grid. E.g.,	if w=50, the width of a	grid cell is w/sample_grid.





example:


First, make map and tree lists; for this example, the lists contain only one filepath each, but in practice they will contain all maps and tree sequences.
```bash
ls tempout/mapnn_123.npy  > tempout/map_list.txt
ls tempout/recap_123.trees  > tempout/tree_list.txt
```

Next, preprocess with `mapNN`:

```bash
python mapnn.py \
       --preprocess \
       --out tempout \
       --simid 1 \
       --seed 1 \
       --n 100 \
       --num_snps 5000 \
       --tree_list tempout/tree_list.txt \
       --target_list tempout/map_list.txt \
       --slim_width 50 \
       --map_width 50 \
       --sample_grid 5
```










### Training

usage:

```bash
python mapnn.py \
       --train \
       --out <str> \
       --seed <int> \
       --n <int> \
       --num_snps <int> \
       --pairs <int> \
       --pairs_encode <int> \
       --validation_split <float> \
       --batch_size <int> \
       --learning_rate <float> \
       --max_epochs <int> \
       --gpu_index <int, or "any">
```

There are some new command line flags, here:

- `train`: tells `mapNN` to train.
- `pairs`: is the number of (random) pairs of individuals to include.
- `pairs_encode`: is the number of pairs to include in the genotype-encoding branch of the network (pairs_encode <= pairs).
- `validation_split`: the proportion of training datasets to hold out for hyperparameter tuning during training.
- `batch_size`: the size of mini batches for stochastic gradient descent.
- `learning_rate`: the starting learning rate; the learning rate is halved every ten epochs without improvement in validation loss.
- `max_epochs`: the maximum number of epochs before training ends.
- `gpu_index`: index of a gpu. Specify "any" to look for any availabel gpu, or "-1" to use CPU (default).


To try out the below command, first download a set of 100 preprocessed datasets from our server:

```bash
wget http://sesame.uoregon.edu/~chriscs/mapNN/Examples.zip
unzip Examples.zip
```

The purpose, here, is to demonstrate that the code runs, not to actually train a "production level" model.
For a real analysis, you will need thousands of training examples, and hours or days of training on a GPU.
In our paper, we used a training set of 50,000.

example:

```bash
python mapnn.py \
       --train \
       --out Examples/Train \
       --seed 1 \
       --n 100 \
       --num_snps 5000 \
       --pairs 450 \
       --pairs_encode 100 \
       --validation_split 0.2 \
       --batch_size 10 \
       --learning_rate 1e-4 \
       --max_epochs 10 \
       --gpu_index -1
```










### Prediction

```bash
python mapnn.py \
       --predict \
       --out <str> \
       --seed <int> \
       --n <int> \
       --num_snps <int> \
       --pairs <int> \
       --pairs_encode <int> \
       --load_weights <str> \
       --batch_size <int> \
       --ranges <float> <float> <float> <float> \
       --gpu_index <int, or "any">
```

Here there are a couple of new flags:

- `predict`: tells `mapNN` to predict
- `load_weights`: path to saved model weights from training
- `ranges`: min and max for disersal prior range, and min and max for density prior range, space-separated (for plotting)

example:

```bash
python mapnn.py \
       --predict \
       --out Examples/Test \
       --seed 1 \
       --n 100 \
       --num_snps 5000 \
       --pairs 450 \
       --pairs_encode 100 \
       --load_weights Examples/Train/mapNN_1_model.hdf5 \
       --batch_size 10 \
       --ranges 0.73,3.08,4,40 \
       --gpu_index -1
```

(The predictions will not look good, since we only did a miniature training run.)





### Empirical analysis

For an empirical analysis, you will likely want to do some new things.
First, ideally we provide a PNG map of the habitat to `mapNN` and SLiM, where habitable areas are black, and unlivable pixels are white.
Second, we will provide the empirical sampling locations to `mapNN` during preprocessing, and sample individuals near the real localities.
The below commands can be used as a starting point for an empirical workflow; although we don't do a full-blown analysis, here.

Cookie cutter the training maps to reflect the empirical habitat

```python
from PIL import Image
from numpy import asarray
import skimage.measure
import numpy as np

# inputs                                                                                                                                            
target = np.load("tempout/mapnn_123.npy")
w = target.shape[0]
data=Image.open("Examples/Empirical/wolf_distribution.png")
temp=asarray(data)
data = np.copy(temp)
outline = data.astype(float)

# compress the mask                                                                                                                                 
rat = int(round(outline.shape[0]/w))
outline = skimage.measure.block_reduce(outline, block_size=(rat,rat,1), func=np.mean)

# apply the cookie cutter:                                                                                                                          
for i in range(w):
    for j in range(w):
        mean_val = np.mean(outline[i,j,0:3])
        if mean_val > (255/2.0): # arbitrary cutoff for grey pixels                                                                                 
            target[i,j,0] = 0.
            target[i,j,1] = 0.

# write
np.save("Examples/Empirical/cookie_123.npy", target)
np.savetxt("Examples/Empirical/cookie_123_disp.csv", target[:,:,0], delimiter=",", fmt='%f')
np.savetxt("Examples/Empirical/cookie_123_dens.csv", target[:,:,1], delimiter=",", fmt='%f')
```




Simulation

```bash
slim -d "MAP_FILE_0='Examples/Empirical/cookie_123_disp.csv'" -d "MAP_FILE_1='Examples/Empirical/cookie_123_dens.csv'" -d "OUTNAME='Examples/Empirical/sim'" -d SEED=2 SLiM_recipes/wolves.slim
```


Preprocessing

```bash
python mapnn.py --preprocess --out Examples/Empirical/ --simid 1 --seed 1 --num_snps 10627 --n 94 --tree_list Examples/Empirical/tree_list.txt --target_list Examples/Empirical/map_list.txt --map_width 50 --slim_width 7109.537608030495 --habitat_map Examples/Empirical/wolf_distribution.png --empirical Examples/Empirical/wolves_n94 --chroms 38
```


Training

```bash
python mapnn.py --train --out Examples/Empirical/ --seed 1 --n 94 --num_snps 10627 --validation_split 0.2 --batch_size 10 --max_epochs 1000 --pairs 650 --pairs_encode 100 --habitat_map Examples/Empirical/wolf_distribution.png
```



Prediction

```bash
python mapnn.py --predict --out Examples/Empirical/ --seed 1 --n 94 --num_snps 10627 --batch_size 10 --pairs 650 --pairs_encode 100 --habitat_map Examples/Empirical/wolf_distribution.png --empirical Examples/Empirical/wolves_n94 --habitat_border Examples/Empirical/wolf_distribution_outline.png --load_weights Empirical/Examples/mapNN_1_model.hdf5 --num_reps 10
```



Predict on bootstrap sims

```bash
python mapnn.py --predict --out Examples/Bootstrap_sims/ --seed 1 --n 94 --num_snps 10627 --batch_size 10 --pairs 650 --pairs_encode 100 --habitat_map Examples/Empirical/wolf_distribution.png --habitat_border Examples/Empirical/wolf_distribution_outline.png --load_weights Empirical/Examples/mapNN_1_model.hdf5
```



Confidence interval command

```bash
python mapnn.py --bootstrap Examples/Empirical/Test_1/mapNN_empirical_pred.npy --out Examples/Bootstrap_sims/ --seed 1 --habitat_map Examples/Empirical/wolf_distribution.png --habitat_border Examples/Empirical/wolf_distribution_outline.png
```










## References

Haller BC, Messer PW. SLiM 4: multispecies eco-evolutionary modeling. The American Naturalist. 2023 May 1;201(5):E127-39.


