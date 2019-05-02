# Installing and Running PyTorch on the HPC Clusters

CSES recommends users follow these directions to install PyTorch on the GPU clusters (Tiger and Adroit).

## Clone the repo

Log in to a head node on one of the GPU clusters (Tiger or Adroit). Then clone the repo to your home directory using:

```
git clone https://github.com/PrincetonUniversity/install_pytorch.git
```

This will get you the file in a folder called `install_pytorch`

## Make an appropriate conda environment

Getting GPU (and some nice MKL) support for PyTorch is as easy as:

```
module load anaconda3
conda create --name pytorch-env pytorch torchvision cudatoolkit=9.0 -c pytorch
```

While we have a newer version of the CUDA toolkit, PyTorch recommends version 9.

Once the command above completes, as long as you have the `anaconda3` module loaded (current session only,
you'll note that we load it in the Slurm script `mnist.cmd`),
you'll have access to `conda` and can use it to access the Python
virtual environment you just created.

Activate the conda environment:

```
conda activate pytorch-env
```

## Running the example

The compute nodes do not have internet access so we must obtain the data in advance. The data can be obtained by

Now that you have the data, you can schedule the job.

Edit the line of `mnist.cmd` to remove the space in the `--mail-user` line
and add your Princeton NetID (or other email address)

Then from the `slurm_mnist` directory run:

```
sbatch mnist.cmd
```

This will request a GPU, 5 minutes of computing time, and queue the job. You should receive a job number and can check if your job is running or queued
via `squeue -u yourusername`.

You'll also receive start and finish emails.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `slurm_mnist` with PyTorch's messages.
