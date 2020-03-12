# Installing and Running PyTorch on the HPC Clusters


[PyTorch](https://pytorch.org) is a popular deep learning library for training artificial neural networks. The installation procedure depends on the cluster:

### Adroit or TigerGPU

```
module load anaconda3
conda create --name torch-env pytorch torchvision cudatoolkit=10.1 --channel pytorch
conda activate torch-env
```

Or maybe you want a few additional packages like matplotlib and tensorboard:

```
module load anaconda3
conda create --name torch-env pytorch torchvision cudatoolkit=10.1 matplotlib tensorboard --channel pytorch
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

### Perseus, Della or TigerCPU

```
module load anaconda3
conda create --name torch-env pytorch torchvision cpuonly --channel pytorch
conda activate torch-env
```

Be sure to include `conda activate torch-env` in your Slurm script.

## Example

The example below shows how to run a simple PyTorch script on one of the clusters. We will train a simple CNN on the MNIST data set. Begin by connecting to a head node on one of the clusters. Then clone the repo:

```bash
$ git clone https://github.com/PrincetonUniversity/install_pytorch.git
$ cd install_pytorch
```

This will create a folder called `install_pytorch` which contains the files needed to run this example. The compute nodes do not have internet access so we must obtain the data while on the head node:

```
$ python download_mnist.py
```

Inspect the PyTorch script called `mnist_classify.py`. Submit the job to the batch scheduler:

```
$ sbatch job.slurm
```

The Slurm script used for the job is below:

```bash
#!/bin/bash
#SBATCH --job-name=torch-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3

conda activate torch-env

srun python mnist_classify.py --epochs=3
```

You can monitor the status of the job with `squeue -u $USER`. Once the job runs, you'll have a `slurm-xxxxx.out` file in the `install_pytorch` directory. This log file contains both PyTorch and Slurm output.

## Multithreading

Even when using a GPU there are still operations carried out on the CPU. Some of these operations have been written to take advantage of multithreading. Try different values for `--cpus-per-task` to see if you get a speed-up:

```
#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=<T>      # cpu-cores per task (>1 if multi-threaded tasks)
```

On TigerGPU, there are seven CPU-cores for every one GPU. Try doing a set of runs where you vary `<T>` from 1 to 7 to find the optimal value.

## GPU Utilization

To see how effectively your job is using the GPU, after submiting the job run the following command:

```
$ squeue -u $USER
```

The rightmost column labeled "NODELIST(REASON)" gives the name of the node where your job is running. SSH to this node:

```
$ ssh tiger-iXXgYY
```

In the command above, you must replace XX and YY with the actual values (e.g., `ssh tiger-i19g1`). Once on the compute node run `watch -n 1 gpustat`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. TensorFlow by default takes all available GPU memory. For this specific example you will see that only about 10% of the GPU cores are utilized. Given that a CNN is being trained on small images (i.e., 28x28 pixels) this is not surprising. You should repeat this analysis with your actual research code to ensure that the GPU is being utilized. For jobs that run for more than 10 minutes you can check utilization by looking at the [TigerGPU utilization dashboard](https://researchcomputing.princeton.edu/node/7171). See the bottom of that page for tips on improving utilization.

Type `Ctrl+C` to exit the `watch` command. Type `exit` to leave the compute node and return to the head node.

## Distributed Training or Using Mulitiple GPUs

If you are getting good GPU utilization then consider using multiple GPUs with [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html). In this case your model will be replicated and fed different batches. Keep in mind that by default the batch size is reduced when multiples GPUs are used. Be sure to use a sufficiently large batch size to keep each GPU busy.

For large models that do not fit in memory, there is the [model parallel](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html) approach. In this case the model is distrbuted over multiple GPUs.

Also take a look at [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [Horovod](https://github.com/horovod/horovod).

## TensorBoard

A useful tool for tracking the progress of PyTorch scripts is [Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html). This can be run on the head node in non-intensive cases. Tensorboard is available via Conda.

## Using PyCharm on TigerGPU

This video shows how to launch PyCharm on a TigerGPU node and use its debugger on a TensorFlow script:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=0XmZsfixAdw" target="_blank">
<img src="http://img.youtube.com/vi/0XmZsfixAdw/0.jpg" alt="PyCharm" width="480" height="270" border="0" /></a>

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
