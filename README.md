
# NeRF for Varying Geometry
The implementation uses PyTorch Lightning for training and provides scripts for dataset generation, training, and rendering.

**Instructions**  
Generate the dataset by running `pix2pix.ipynb`. This notebook handles the preprocessing and creation of the training dataset with outfit codes.  
Train the NeRF model by running `train_nerf.ipynb`. This notebook sets up the training pipeline and calls the necessary modules.

**Code Structure**  
`person.py` contains the custom dataset loader for the project, including functionality for outfit code management.  
`nerf.py` implements the NeRF model class. This model is integrated into the training pipeline via the `NeRFSystem` class in `train.py`.  
`rendering.py` includes the `render_ray` functions, which handle the ray tracing and volume rendering processes, essential for generating high-quality NeRF renders.  
`train.py` configures the training pipeline using PyTorch Lightning, managing the training loop and checkpoints.

**Training Framework**  
The model is implemented and trained using [PyTorch Lightning](https://www.pytorchlightning.ai/), which simplifies distributed training and provides tools for efficient experimentation.

**Acknowledgments**  
This project builds upon the NeRF model architecture: https://github.com/kwea123/nerf_pl
