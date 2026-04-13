# ResponseRequirements
## Overview
This repository contains codes for studying the impact of spacecraft orbital uncertainty on LISA TDI outputs.

![LISA orbits](report/figures/3d_orbit_around_sun.png)

## Installation
Follow these steps to set up the project on your local machine:


1. **Clone the repository**:
    ```bash
    git clone https://github.com/lorenzsp/ResponseRequirements.git
    cd ResponseRequirements
    ```

2. **Install dependencies**:
    Ensure you have conda installed. Then run:
    ```bash
    conda create -n lisa_resp -c conda-forge gcc_linux-64 gxx_linux-64 python=3.12
    conda activate lisa_resp
    pip install lisaorbits healpy pytdi tqdm numpy Cython scipy jupyter ipython h5py matplotlib tqdm
    ```

## Usage
Execute the following command to run the analysis:
```bash
bash run_segwo_analysis.sh
```


## Contact
For questions or feedback, please contact [Lorenzo Speri](https://github.com/lorenzsp).