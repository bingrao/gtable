
# Code arch
```
├── .run/
├── docs/
├── examples/
├── gtable/
├── .gitignore
├── gtable.yml
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── generation.py
└── evaluate.py
```

* **.run/**  : This directory holds shell scripts and Pycharm run configurations for training and testing.
* **docs/** :  The documents provided by this project
* **examples/** : A list of datasets, and their correpsonding YML configurations files. The output of models also will be saved into the corresponding dataset folder. 
* **gtable/**  : The directory contains source code, but now there are subdirectories. 
* **.gitignore** : The configuration file for git command
* **gtable.yml** : A conda config file for this project
* **LICENSE** : The license for this project
* **Makefile** : A makefile to compile this project  
* **README.md** : A readme file
* **requirements.txt** : A list of required python packages
* **generation.py** : A python run file for  training and generate process
* **evaluate.py** : A python run file for evaluating performance between real and fake datasets



# Software Package Installation and Configuration

1. Install CUDA and GPU Driver if using GPU (We are using cuda-10.1)

2. Install latest miniconda as Python virtual environment management
   
3. Create a conda environment using a YML config file:
```
conda env create --name "gtable" -f gtable.yml

or 

make
```

4. Activate Python Virtual Environment
```
conda activate gtable
```
5. Generate dependency yaml file:
```
conda env export > gtable.yml
```

A few other frequently used commands
```
# list all the conda environment available
conda info --envs  
# Create new environment named as `envname`
conda create --name envname
# Remove environment and its dependencies
conda remove --name envname --all
# Clone an existing environment
conda create --name clone_envname --clone envname
```

# How to run training process

```
python generation.py --help
python generation.py -config examples/adults/config/adult_1_1.yml
```

# How to run evaluate process
```
python evaluate.py --help
python evaluate.py -config examples/adults/config/adult_1_1.yml
```
