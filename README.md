# Installing and Running PyTorch on the HPC Clusters


[PyTorch](https://pytorch.org) is a popular deep learning library for training artificial neural networks. The installation procedure depends on the cluster:

### Adroit or TigerGPU

```
module load anaconda3
conda create --name torch-env pytorch torchvision cudatoolkit --channel pytorch
conda activate torch-env
```

Be sure to include `conda activate torch-env` and `#SBATCH --gres=gpu:1` in your Slurm script.

### Traverse

```
module load anaconda3
conda create --name=torch-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda pytorch
conda activate torch-env
# accept the license agreement if asked
```

Be sure to include `conda activate torch-env` and `#SBATCH --gpus-per-node=1` in your Slurm script.

Note that the `torchvision` package is not presently available for the ppc64le architecture. If you need datasets or models from this package you will need to install it on another cluster and then transfer the files to Traverse.

### Perseus, Della or TigerCPU

```
module load anaconda3
conda create --name torch-env pytorch torchvision -c pytorch
conda activate torch-env
```

Be sure to include `conda activate torch-env` in your Slurm script.

# Example

The example below shows how to run a simple PyTorch script on one of the clusters.

## Clone the repo

Log in to a head node on one of the clusters. Then clone the repo using:

```
git clone https://github.com/PrincetonUniversity/install_pytorch.git
```

This will create a folder called `install_pytorch` which contains the files needed to follow this tutorial.

## Make a conda environment and install

Next we create a conda environment that includes PyTorch and its dependencies (note that you may consider replacing the environment name "torch-env" with something more specific to your work):

```
# adroit or tigergpu
module load anaconda3
conda create --name torch-env pytorch torchvision cudatoolkit --channel pytorch
```

Once the command above completes, as long as you have the `anaconda3` module loaded (current session only,
you'll note that we load it in the Slurm script `mnist.slurm`),
you'll have access to `conda` and can use it to access the Python virtual environment you just created.

Activate the conda environment:

```
conda activate torch-env
```

Let's make sure our installation can find the GPU by launching an interactive session on one of the compute nodes:

```
salloc -t 00:05:00 --gres=gpu:1
```

When your allocation is granted, you'll be moved to a compute node. Execute the following command on the compute node to test for GPU support:

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

```

If the output is "True" and "Tesla P100-PCIE-16GB" (on Tiger) or "Tesla V100-SXM2-32GB" (on Traverse) then your installation of PyTorch is GPU-enabled. Type `exit` to return to the head node.


## Running the example

The compute nodes do not have internet access so we must obtain the data in advance. Run the `mnist_download.py` script from the `install_pytorch` directory on the head node:

```
python mnist_download.py
```

Now that you have the data, you can schedule the job using the following command:

```
sbatch mnist.slurm
```

This will request one GPU, 5 minutes of computing time, and queue the job. You should receive a job number and can check if your job is running or queued
via `squeue -u <your-username>`.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `install_pytorch` directory. This log file contains both Slurm and PyTorch messages.

## Examining GPU utilization

To see how effectively your job is using the GPU, immediately after submiting the job run the following command:

```
squeue -u $USER
```

The rightmost column labeled "NODELIST(REASON)" gives the name of the node where your job is running. SSH to this node:

```
ssh tiger-iXXgYY
```

Once on the compute node run `watch -n 1 gpustat`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. For this specific example you will see that only about 10% of the GPU cores are utilized. Given that a CNN is being trained on small images (i.e., 28x28 pixels) this is not surprising. You should repeat this analysis with your actual research script to ensure that your GPUs are nearly fully utilized.

Type `Ctrl+C` to exit the `watch` screen. Type `exit` to return to the head node.

## More examples

More PyTorch example scripts are found here: [https://github.com/pytorch/examples](https://github.com/pytorch/examples)

If you encounter any difficulties while installing PyTorch on one of our HPC clusters then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.
