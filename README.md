# Autopoietic-Emergent-Scheduler
SUPSI+LAKE WP4
## Requirements

- [Python 3.11.7 or higher]
- [Numpy 1.26.4 or higher]
- [Mesa 2.2.4]
- [Other Python libraries listed in the scripts, last stable version]

## Deploy
### Instructions
1. In the same directory where all the files of this library are stored, the following directory should also be created: "Data/Swarming/".
2. Line 32 of "main_swarming.py" must point to the directory where all the files of this library are stored.
3. Parameters of the problem to be solved: gamma in "config0.yaml" and in "config1.yaml" (the value must be the same in the two files); scheduler and queue parameters inside "utilities_swarming.py"; optimization parameters in "main_swarming.py".
4. Run Python script 'main_swarming.py' to launch the D-GLIS algorithm for solving the distributed Swarming Algorithm calibration problem (e.g., mpirun -n 2 python -u main_swarming.py).


## To Read
- https://ieeexplore.ieee.org/abstract/document/10107979
- https://ieeexplore.ieee.org/document/10705177
