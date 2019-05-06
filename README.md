# Installing and Running PyTorch on the HPC Clusters

CSES recommends users follow the directions below to install PyTorch on the GPU clusters (Tiger and Adroit).

## Clone the repo

Log in to a head node on one of the GPU clusters (Tiger or Adroit). Then clone the repo to your home directory using:

```
git clone https://github.com/PrincetonUniversity/install_pytorch.git
```

This will create a folder called `install_pytorch` which contains the files needed to follow this tutorial.

## Make a conda environment and install

Next we create a conda environment that includes pytorch and its dependencies. Note: You may consider replacing the environment name "myenv" with something more specific to your work.

```
module load anaconda3
conda create --name myenv pytorch torchvision cudatoolkit=9.0 -c pytorch
```

While we have a newer version of the CUDA toolkit installed on the HPC clusters, PyTorch recommends version 9.

Once the command above completes, as long as you have the `anaconda3` module loaded (current session only,
you'll note that we load it in the Slurm script `mnist.slurm`),
you'll have access to `conda` and can use it to access the Python
virtual environment you just created.

Activate the conda environment:

```
conda activate myenv
```

Let's make sure our installation can find the GPU:

```
salloc -N 1 -n 1 -t 5:00 --gres=gpu:1
```

When your allocation is granted, you'll be moved to a compute node. Execute the following commond on the compute to test for GPU support:

```
python -c "import torch; print(torch.cuda.is_available(); print(torch.cuda.get_device_name(0)))"

```

If the output is "True" and "Tesla P100-PCIE-16GB" (on tiger) then your installation of PyTorch can use GPUs.


## Running the example

The compute nodes do not have internet access so we must obtain the data in advance. The data can be obtained by running mnist_download.py from the `install_pytorch` directory on the head node:

```
python mnist_download.py
```

Now that you have the data, you can schedule the job. From the `install_pytorch` directory run:

```
sbatch mnist.slurm
```

This will request one GPU, 5 minutes of computing time, and queue the job. You should receive a job number and can check if your job is running or queued
via `squeue -u <yourusername>`.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `install_pytorch` directory. This log file contains both Slurm and PyTorch messages.

## Examining GPU utilization

To see how effectively your job is using the GPU, immediately after submiting the job run the following commond:

```
squeue -u <username>
```

The rightmost column gives the name of the node where your job is running. SSH to this node:

```
ssh tiger-iXXgYY
```

Once on the compute node run `watch -n 1 gpustat`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. For this specific example you will see that the GPU is not effectively used with the value fluctuating around 10%. Given that a CNN is being trained on small images (i.e., 28x28 pixels) this is not surprising. Be sure to repeat this analysis with your actual research script.

## More examples

More PyTorch example scripts are found here:
```
https://github.com/pytorch/examples
```

Please send questions/issues to cses@princeton.edu.
