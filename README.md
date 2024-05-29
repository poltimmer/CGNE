# CGNE
Repository for the paper Efficient Probabilistic Modeling of Crystallization at Mesoscopic Scale: https://arxiv.org/abs/2405.16608

## Requirements
### CGNE
- Python 3.10
- PyTorch
- Lightning
- NumPy
- H5Py
- PyYAML (for configuration files)
- Wandb (for logging)
- einops (for logging)
- POT (for calculation of Wasserstein distance)

### LCA
- CUDA compatible GPU
- Python 3.10
- NumPy
- Numba
- H5Py
- PyYAML
- Matplotlib


## Usage
### CGNE
To train the model, run the following command:
```bash
python -m CGNE.train --config config.yaml
```
where `config.yaml` is the configuration file. A configuration file with default values is provided in `configs/default.yaml`.

Any configuration parameter can be overridden by adding it to the configuration file. For example:
```yaml
model:
  z_dim: 128
data:
  batch_size: 64
```

### LCA
To run the LCA simulation, run the following command:
```bash
python -m LCA.main
```
This accepts command line arguments, as well as a configuration file.

## Citation
If you use this code, please cite the following paper:
```bibtex
@misc{timmer2024efficient,
      title={Efficient Probabilistic Modeling of Crystallization at Mesoscopic Scale}, 
      author={Pol Timmer and Koen Minartz and Vlado Menkovski},
      year={2024},
      eprint={2405.16608},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

