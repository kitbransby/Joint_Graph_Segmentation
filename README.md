# Joint Dense-Point Representation for Contour-Aware Graph Segmentation

### Installation 

We use the same installation process as [HybridGNet](https://github.com/ngaggion/HybridGNet), and these dependencies can be installed in a fresh conda environment using:

```conda env create -f environment.yml```

To train and evaluate the Rasterize model, a differentiable rasterization pipeline is required, which can be installed by 
following install instructions for [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer. 
). We advise that this is created in a separate environment. 

### Datasets

Instructions for download and preprocess datasets can be found in `Datasets/README.md`

### Training

To train our joint dense-point network from scratch with a HCD loss on the JSRT & Padchest dataset, run the following command:

```python Train/trainerLH_Joint_HCD.py```

Trainers for all models and baselines are available in `Train/`, where LH (Lungs & Heart) = JSRT & Padchest dataset, and L (Lungs) = Montgomery & Shenzen dataset.
Training weights will be saved to `Results/` dir. 

### Paper Reproducibility 

To reproduce the results in the paper, first download the model weights [here](https://drive.google.com/drive/folders/1V-Ohw1_EbIr4rBnk7h8OgRpvWE-JeyJ4?usp=sharing), and place them in the relevant `weights/`
directory. To evaluate the models, run the evaluation scripts in `Evaluate/`, making sure that you have created the directories as described in `Evaluate/README.md`

