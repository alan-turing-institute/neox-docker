# neox-docker

GPT Neox docker builder

This repository contains the submodules and patches needed to allow GPT Neox to run using nvcr.io/nvidia/pytorch:22.12-py3 on Isambard-Ai.

## Apply the patches

The patches can be applied as follows. For triton:
```
pushd triton
git am ../patches/triton/*.patch
popd
```

For gpt-neox:
```
pushd gpt-neox
git am ../patches/gpt-neox/*.patch
popd
```

## Remove the patches

In the root folder, run:
```
git submodule update --init
```

## Patch generation

If you want to update the patches with new content, the process is to first apply the existing patches, then to make your changes, then to generate the patches.
After that you should be safe to rest the submodules to their original state (because you can then re-apply your patches if you want to).

To demonstrate how this is done we'll use gpt-neox as an example, but the same process applies for triton as well.

First apply the existing patches as described above.

After making and committing your changes to the submodule, generate new patches.
You'll need to change the `2` at the end to be the total number of patches to output (which will be the number of previous patches plus the number of new commits you've made):
```
git format-patch -o ../patches/gpt-neox -N --no-signature --zero-commit HEAD~2
popd
```

You can now reset the repository to its original state as described in the previous section.

## Licence

The code in this repository (not the submodules) is licensed under the BSD 2-Clause licence.


