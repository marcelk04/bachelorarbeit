# NeRF and 3DGS based reconstruction of direct-global separated multiview image data

<img src="comparison.svg">

## Cloning the Repository

Make sure you clone the repository along with all submodules by using:

```bash
git clone --recurse-submodules https://github.com/marcelk04/bachelorarbeit.git
```

Afterwards, navigate to the directory with:

```bash
cd bachelorarbeit
```

## Installing the dependencies

Create the environment and install all necessary packages using:

```bash
conda env create -f environment.yml
```

You can then activate the environment with:

```bash
conda activate ba
```

If you have any dependency problems with the environment above, you can also try:

```bash
conda env create -f environment_alternative.yml
```

This environment specifies the exact version for all packages used.

### Building Instant-NGP

For detailed instructions, check out [the Instant-NGP README](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux). All required packages are already installed on the VCI GPU server. The essential steps for building are layed out below.

First, make sure the conda environment is activated, as the python bindings have to be compiled for the correct python version.

```bash
conda activate ba
```

To start with building, navigate to the Instant-NGP directory:

```bash
cd submodules/instant-ngp
```

Then, use CMake to generate a build configuration:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

And finally build the project:

```bash
cmake --build build --config RelWithDebInfo -j
```

Afterwards, you should be done and all scripts should run without any problems.

## Acquiring Datasets

You can find some datasets under the release page.
They were rendered from `marcus_light_sphere_hair.xml`, each with a different number of training cameras.

If you are interested in rendering yourself, you will first need to download the [head model](https://www.digitalrealitylab.com/sample-model) and [hair model](https://www.cemyuksel.com/research/hairmodels).
The head model used was the `Head PBR Scan Sample` and the hair model is `wWavy`.
Place the extracted folder containing the head model (name should be something like `Marcus_PBR_Sample_01`) and place it in a folder called `head_models`.
Similarily, place the `*.hair`file in a folder called `hair_models`.
To convert the hair strands to B-spline curves for Mitsuba, use:

```bash
python src/data_generation/hair_to_curves.py -i hair_models/wWavy.hair -o hair_models/wWavy.txt
```

Afterwards, you are ready to render your dataset. The required script `src/data_generation/generate_images.py` is already included in each `train_*.sh` script, so you just have to uncomment the line.

## Reconstructing a Dataset

Simply run the `train_*.sh` script of your choice, e.g.:

```bash
bash train_3dgs_tandem.sh
```

Make sure that the paths in the file match your desired dataset and output directory.