# neox-docker

GPT Neox docker builder

This repository contains the submodules and patches needed to allow GPT Neox to run using nvcr.io/nvidia/pytorch:22.12-py3 on Isambard-Ai.

See the [Bristol Centre for Supercomupting Docs](https://docs.isambard.ac.uk/user-documentation/guides/containers/podman-hpc/) for how to make use of `podman-hpc` on Isambard-AI.

## Building the Docker image on Isambard-AI

In all of the commands below, we use an image tag of `llewelld/isambard-ai-neogx:v1.3`.
You should change this to something more appropriate for your needs.

To build the Docker image on Isambard-AI for use with `podman-hpc` you can use the following steps.
```
$ git clone https://github.com/llewelld/neox-docker.git --recurse-submodules
$ pushd neox-docker
$ podman-hpc build \
    --build-arg mpi_type=single \
    --build-arg feature_branch=main \
    -t llewelld/isambard-ai-neogx-single:v1.7 .
$ popd
```

If you're working on a separate feature branch, replace `main` in the above `build` command with the name of your branch.

You can also pull the image directly from docker hub.
```
$ podman-hpc pull llewelld/isambard-ai-neogx:v1.3
```

You'll need to then migrate the image to make it available on other compute nodes.
```
$ podman-hpc migrate llewelld/isambard-ai-neogx:v1.3
```

## Cross building the Docker image

We recommend you build the image on Isambard (see previous section) but it also possible to build it locally for ARM using cross compilation with Docker.

The following commands will do this.
As before you should change the tag to something more appropriate for your needs.

```
$ git clone https://github.com/llewelld/neox-docker.git --recurse-submodules
$ pushd neox-docker
$ docker buildx build \
    --build-arg mpi_type=single \
    --build-arg feature_branch=main \
    --platform linux/arm64 \
    -t llewelld/isambard-ai-neogx:v1.3 .
$ popd
```
Once again, if you're working on a separate feature branch, replace `main` in the above `build` command with the name of your branch.

## Using the image

To make use of the images you'll need to create `experiment`, `jobs` and `neox_models` directories to map inside the container.

Note also that you'll need to generate a hostfile so that OpenMPI knows where to communicate with.
The `scripts/write_hostfile.sh` can be used to do this.
This creates a file called `/tmp/hostfiles/hosts_$SLURM_JOBID/for_deepspeed.txt` where the job ID is used to avoid clashes when running multiple jobs simultaneously on different sets of nodes.
In the commands below we then map this to `/hosts/for_deepspeed.txt` inside the container.
The `DLTS_HOSTFILE` environment variable is set to point to this location to ensure it gets picked up automatically.

In the commands below we assume your working directory contains a cloned copy of this `neox-docker` repository, along with the `experiment`, `jobs` and `neox_models` directories.

```
$ srun --time 2:00:00 --gpus=4 --mpi=pmix --pty --nodes=1 --ntasks-per-node=4 /bin/bash
$ module load libfabric
$ ./neox-docker/scripts/write_hostfile.sh
$ SCRATCH=${HOME/home/scratch}
$ podman-hpc run --mpi-trial -it --gpu \
    -e TMPDIR -v $TMPDIR \
    -v ./experiment/:/experiment \
    -v ./jobs/:/jobs \
    -v ./neox_models/:/neox_models \
    -v /tmp/hostfiles/hosts_${SLURM_JOBID}/:/hosts/ \
    -e MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
    -e MASTER_PORT=12802 \
    --entrypoint /bin/bash llewelld/isambard-ai-neogx:v1.3
```

Once inside the container, run the training script as follows (directed at your configuration file).

```
$ source /host/adapt.sh
$ python deepy.py train.py /experiment/jobs/01_poison_71000/configs/6.9B-deduped.yml
```

## Editing the Dockerfile

If you want to make changes to the Docker build process you'll need to edit the Dockerfile and may also need to amend the patches that are applied to `gpt-neox` and `triton`.

The steps for updating the patches involve a four-stage process:
1. Remove any existing patches.
2. Apply the existing patches.
3. Make the changes you want to make.
4. Regenerate the patches to include your changes.

Having done so you can then commit your changes (including the changes to the patches) to a feature branch.

The sections below explain how each of these can be achieved.

### Remove any existing the patches

In the root folder, run:
```
$ git submodule update --init
```

### Apply the existing patches

The patches can be applied as follows.
For gpt-neox:
```
$ pushd gpt-neox
$ git am ../patches/gpt-neox/*.patch
$ popd
```

For triton:
```
$ pushd triton
$ git am ../patches/triton/*.patch
$ popd
```

### Patch generation

If you want to update the patches with new content, the process is to first apply the existing patches, then to make your changes, then to generate the previous patches again alongside your new patches.
Once you've generated your patches it should be safe to reset the submodules to their original state, because you can then re-apply your patches if you want to.

To demonstrate how this is done we'll use gpt-neox as an example, but the same process applies for triton as well.

First apply the existing patches as described above.

After making and committing your changes to the submodule, generate new patches.
You'll need to change the `2` at the end to be the total number of patches to output (which will be the number of previous patches plus the number of new commits you've made):
```
$ git format-patch -o ../patches/gpt-neox -N --no-signature --zero-commit HEAD~2
```

You can now reset the repository to its original state as described in the previous section.

## Licence

The code in this repository (not the submodules) is licensed under the BSD 2-Clause licence.

## Credits

Thanks to all of the following who have contributed to this repo (in no particular order):

1. Ed Chapman (@edchapman88), The Alan Turing Institute.
2. Iain Stenson (@Iain-S), The Alan Turing Institute.
3. Alexandra Souly (@alexandrasouly-aisi), AI Safety Institute.
4. Wahab Kawafi (@wahabk), BriCS.
5. David Llewellyn-Jones (@llewelld), The Alan Turing Institute.

