# Anti-Adversarial Networks: Gradients 

This is a WIP package for anti-adversarial network exploration as part of a thesis in Mathematics at the University of Waterloo. This README will be updated as experiments and code progresses

**No code are guaranteed at this stage**

# Installation & Running 

This package assumes python `3.9` or greater, though it may work on lower versions. 

## Virtual Environment

It is highly recommended to use a virtual environment to run any code. This can be done through various methods:

1. [virtualenv](https://virtualenv.pypa.io/en/latest/) (recommended): `$ python3 -m virtualenv --python=<path/to/python/3.9+> <path/to/venv/directory> && source ~/<path/to/venv/name/bin/activate`
2. [venv](https://docs.python.org/3/tutorial/venv.html) (virtualenv wrapper): `$ python3 -m <path/to/venv/dir> <venv_name0 && source ~/<path/to/venv/name/bin/activate`>
3. [conda](https://docs.conda.io/en/latest/) (recommended on Windows): `$ conda create --name adv python=3.9 && conda activate adv`


## Packages & Jupyter Notebook

Most experiments are shown in `experiments/notebooks/*.ipynb`, which can be run using [Jupyter Notebooks](www.jupyter.org)

```
(venv) $ python -m pip install -r src/requirements.txt
(venv) $ python -m pip install -e /src/adversarial
```

## Adv package

This package is not hosted on PyPi. It can be installed localled by as an enditable package into your python interpreter's path, assuming `cwd` is the anti adversarial project root:

`(venv) $ pip install -e /src/adversarial`

This package contains the Network presets and helper code such as plotting and metrics. 

## GPU 

[Pytorch](https://pytorch.org/) may be configured for CPU or GPU based training/inference, please refer to their [getting-started guide](https://pytorch.org/get-started/locally/) to see how to install torch on your system for use with or without gpu access. 

**Note:** Post-installation `setup` of the adversarial package of cuda 10.2/11.1 has not yet been tested. 


# Structure

```
.
├── README.md                   # Project Running Instructions and Disambiguation
├── data                        # Shared Data 
│   ├── ... 
├── experiments                 # Individual Experiments 
│   └── exp_name                # Experiment Structure Template
│       ├── code
│       ├── data                # Experiment-unique Data
│       ├── notebooks
│       ├── results             # Static Results (plots et c)
│       └── scripts
├── notebooks                   # Example Notebooks
│   ├── ... 
├── requirements.txt            # Global/Experiments Requirements
└── src
    └── adversarial             # Adversarial Package, pip-installable
        ├── README.md
        ├── adversarial
        ├── ...       
        ├── pyproject.toml      # Adversarial Package Requirements (auto-installed)
        ├── setup.py
        └── tests
```
