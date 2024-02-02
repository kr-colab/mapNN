# mapNN
Neural network for estimating demographic maps from SNPs




### Install

```
conda create -n mapnn python=3.9 --yes
conda activate mapnn
pip install --upgrade pip
pip install -r requirements/development.txt
```

Test command:

```
wget http://sesame.uoregon.edu/~chriscs/mapNN/Examples.zip
unzip Examples.zip 
python mapnn.py --train --out Examples/Example_data/ --seed 123 --num_snps 5000 --n 100
```

GPUs
```
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

```
(mapnn) chriscs@poppy:~/Software/mapNN$ python mapnn.py --train --out Examples/Example_data/ --seed 123 --num_snps 5000 --n 100 --gpu any
```

And simultaneously run `nvidia-smi` or `gpustat` to make sure the GPU is firing.
