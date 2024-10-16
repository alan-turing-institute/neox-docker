# neox-docker

GPT Neox docker builder

This repository contains the submodules and patches needed to allow GPT Neox to run using nvcr.io/nvidia/pytorch:22.12-py3 on Isambard-Ai.

See the [Bristol Centre for Supercomupting Docs](https://docs.isambard.ac.uk/user-documentation/guides/containers/podman-hpc/) for how to make use of `podman-hpc` on Isambard-AI.

## Credits

The changes to get this to work were put together by Ed Chapman (@edchapman88) and Iain Stenson (@Iain-S) from the Alan Turing Institute.

## Building the Docker image

To build the image locally, you'll need to cross-build it for ARM.
The following should work (you'll probably want to change the tag):

```
$ git clone https://github.com/llewelld/neox-docker.git --recurse-submodules
$ cd neox-docker
$ docker buildx build -t llewelld/isambard-ai-neogx:v1.3 --platform linux/arm64 .
```

To build it on Isambard-AI is similar, but no cross-compilation is needed and you should use `podman-hpc` rather than `docker`.
The following should work:
```
$ git clone https://github.com/llewelld/neox-docker.git --recurse-submodules
$ cd neox-docker
$ podman-hpc build -t llewelld/isambard-ai-neogx:v1.3 .
```

You can also pull the image directly from docker hub.
```
$ podman-hpc pull llewelld/isambard-ai-neogx:v1.3
```

Either way, on Isambard-AI you'll need to migrate the image to make it available on the compute nodes.
```
$ podman-hpc migrate localhost/llewelld/isambard-ai-neogx:v1.4
```

You're now ready to run the image on a compute node.
```
$ srun --time 2:00:00 --gpus=4 --pty /bin/bash
$ podman-hpc run -it \
    -e TMPDIR -v $TMPDIR \
    -v ./gpt-neox/jobs/:/jobs \
    -v ./neox_models/:/neox_models \
    --gpu \
    --entrypoint /bin/bash \
    --ipc=host \
    llewelld/isambard-ai-neogx:v1.3
```

## Apply the patches

The patches can be applied as follows.
For triton:
```
$ pushd triton
$ git am ../patches/triton/*.patch
$ popd
```

For gpt-neox:
```
$ pushd gpt-neox
$ git am ../patches/gpt-neox/*.patch
$ popd
```

## Remove the patches

In the root folder, run:
```
$ git submodule update --init
```

## Patch generation

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


