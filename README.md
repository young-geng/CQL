# SimpleSAC
A simple and modular implementation of the [Soft Actor Critic](https://arxiv.org/abs/1812.05905) algorithm in PyTorch.


## Installation

1. Install and use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate SimpleSAC
```
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

2. Add this repo directory to your `PYTHONPATH` environment variable or pip install this repo with:
```
pip install -e .
```

## Run Experiments
You can run experiments using the following command:
```
python -m SimpleSAC.main --env 'HalfCheetah-v2' --output_dir './experiment_output' --device='cuda'
```
If you want to run on CPU only, just omit the `--device='cuda'` part.
All available command options can be seen in SimpleSAC/main.py.


## Visualize Experiments
You can visualize the experiment metrics with viskit:
```
python -m viskit './experiment_output'
```
and simply navigate to [http://localhost:5000/](http://localhost:5000/)


## Credits
The project organization is inspired by [TD3](https://github.com/sfujim/TD3).
The SAC implementation is based on [rlkit](https://github.com/vitchyr/rlkit).
The viskit visualization is taken from [viskit](https://github.com/vitchyr/viskit), which is taken from [rllab](https://github.com/rll/rllab).

