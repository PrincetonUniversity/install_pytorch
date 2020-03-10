# Installing and Running PyTorch on the HPC Clusters


[PyTorch](https://pytorch.org) is a popular deep learning library for training artificial neural networks. The installation procedure depends on the cluster:

### Adroit or TigerGPU

```
module load anaconda3
conda create --name torch-env pytorch torchvision cudatoolkit=10.1 --channel pytorch
conda activate torch-env
```

Be sure to include `conda activate torch-env` and `#SBATCH --gres=gpu:1` in your Slurm script.

### Traverse

```
module load anaconda3
conda create --name=torch-env --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda pytorch torchvision
conda activate torch-env
# accept the license agreement if asked
```

Be sure to include `conda activate torch-env` and `#SBATCH --gpus-per-node=1` in your Slurm script.

Note that the `torchvision` package is not presently available for the ppc64le architecture. If you need datasets or models from this package you will need to install it on another cluster and then transfer the files to Traverse.

### Perseus, Della or TigerCPU

```
module load anaconda3
conda create --name torch-env pytorch torchvision cpuonly --channel pytorch
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
conda create --name torch-env pytorch torchvision cudatoolkit=10.1 --channel pytorch
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

Once on the compute node run `watch -n 1 nvidia-smi`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. For this specific example you will see that only about 10% of the GPU cores are utilized. Given that a CNN is being trained on small images (i.e., 28x28 pixels) this is not surprising. You should repeat this analysis with your actual research script to ensure that your GPUs are nearly fully utilized.

Type `Ctrl+C` to exit the `watch` screen. Type `exit` to return to the head node.

[View](https://researchcomputing.princeton.edu/node/7171) the GPU utilization dashboard for TigerGPU.


## Using Multiple GPUs

If you are getting good GPU utilization then consider using multiple GPUs with [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html). In this case your model will be replicated and fed different batches. Keep in mind that by default the batch size is reduced when multiples GPUs are used. Be sure to use a sufficiently large batch size to keep each GPU busy.

For large models that do not fit in memory, there is the [model parallel](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html) approach. In this case the model is distrbuted over multiple GPUs.

Also take a look at [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [Horovod](https://github.com/horovod/horovod).


## TensorBoard

A useful tool for tracking the progress of PyTorch scripts is [Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html). This can be run on the head node in non-intensive cases.

## More examples

More PyTorch example scripts are found here: [https://github.com/pytorch/examples](https://github.com/pytorch/examples)

## How to Learn PyTorch

See the [material](https://github.com/Atcold/pytorch-Deep-Learning) and companian webiste ([English](https://atcold.github.io/pytorch-Deep-Learning/) and [Chinese](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/docs/zh/README-ZH.md)) of Prof. [Alf Canziani](https://twitter.com/alfcnz?lang=en) of NYU.

There is also a free book [here](https://pytorch.org/deep-learning-with-pytorch).

## Where to Store Your Files

You should run your jobs out of `/scratch/gpfs/<NetID>` on the HPC clusters. These filesystems are very fast and provide vast amounts of storage. **Do not run jobs out of `/tigress` or `/projects`. That is, you should never be writing the output of actively running jobs to those filesystems.** `/tigress` and `/projects` are slow and should only be used for backing up the files that you produce on `/scratch/gpfs`. Your `/home` directory on all clusters is small and it should only be used for storing source code and executables.

The commands below give you an idea of how to properly run a PyTorch job:

```
$ ssh <NetID>@tigergpu.princeton.edu
$ cd /scratch/gpfs/<NetID>
$ mkdir myjob && cd myjob
# put PyTorch script and Slurm script in myjob
$ sbatch job.slurm
```

If the run produces data that you want to backup then copy or move it to `/tigress`:

```
$ cp -r /scratch/gpfs/<NetID>/myjob /tigress/<NetID>
```

For large transfers consider using `rsync` instead of `cp`. Most users only do back-ups to `/tigress` every week or so. While `/scratch/gpfs` is not backed-up, files are never removed. However, important results should be transferred to `/tigress` or `/projects`.

The diagram below gives an overview of the filesystems:

![tigress](https://tigress-web.princeton.edu/~jdh4/hpc_princeton_filesystems.png)

## Getting Help

If you encounter any difficulties while installing PyTorch on one of our HPC clusters then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.
