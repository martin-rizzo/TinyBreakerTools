# tbmake (TinyBreaker Make)

`tbmake` is a command-line tool designed to create TinyBreaker models. These models combine the strengths of a PixArt model with a refiner model (either SD1.5 or SDXL) to generate high-quality images.

## Overview

TinyBreaker models are built by merging a pre-trained PixArt model with a refiner model. This combination allows for efficient and effective image generation, leveraging the strengths of both architectures. The `tbmake` tool simplifies this process, allowing users to create their custom TinyBreaker models by specifying the desired models and parameters.

## Features

english: 

 - **Support PixArt model resolutions:** Allows you to integrate pixart models of different resolutions 512px, 1024px, 2048px
 - **Flexible Refiner Selection:** Supports both SD1.5 and SDXL refiner models.
 - **Simple Command-Line Interface:** Easy to use with clear command-line arguments.

## Installation

Currently, `tbmake` is a command-line tool and does not require installation via pip. Simply clone the repository and ensure you have the required dependencies (e.g., PyTorch, Transformers, etc.) installed.

```bash
  git clone https://github.com/martin-rizzo/TBMake
  cd TBMake
  # create the python virtual environment with the dependencies of the project
  ./tbmake.sh --create-venv
```

## Usage

The `tbmake` command takes several arguments to create a TinyBreaker model. Here's how to use it:

### SD1.5 as TinyBreaker Refiner

```bash
tbmake --sd <sd_model>.safetensors --pixart <pixart-model>.safetensors --support support-models.safetensors --resolution 1024
```

-   `--sd <sd_model>.safetensors`: Specifies the path to the SD1.5 refiner model in `.safetensors` format.
-   `--pixart <pixart-model>.safetensors`: Specifies the path to the PixArt model in `.safetensors` format.
-   `--support support-models.safetensors`: Specifies the path to the support models in `.safetensors` format. **This file is provided with the `tbmake` project.**
-   `--resolution <resolution>`: Declares the resolution for which the PixArt model was trained (e.g., 1024).

### SDXL as TinyBreaker Refiner

```bash
tbmake --sdxl <sdxl_model>.safetensors --pixart <pixart-model>.safetensors --support support-models.safetensors --resolution 1024
```

-   `--sdxl <sdxl_model>.safetensors`: Specifies the path to the SDXL refiner model in `.safetensors` format.
-   `--pixart <pixart-model>.safetensors`: Specifies the path to the PixArt model in `.safetensors` format.
-   `--support support-models.safetensors`: Specifies the path to the support models in `.safetensors` format. **This file is provided with the `tbmake` project.**
-   `--resolution <resolution>`: Declares the resolution for which the PixArt model was trained (e.g., 1024).

## Example

```bash
tbmake --sdxl sdxl-base-1.0.safetensors --pixart pixart-alpha-base.safetensors --support support-models.safetensors --resolution 1024
```

This command will create a TinyBreaker model using `sdxl-base-1.0.safetensors` as the SDXL refiner, `pixart-alpha-base.safetensors` as the PixArt model, the provided `support-models.safetensors` and set the output resolution to 1024.

## Support Models

The `support-models.safetensors` file is crucial for the proper functioning of TinyBreaker models. This file contains pre-trained support models that are necessary for the merging process. **This file is already included in the repository.**

## License

Copyright (c) 2025 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.

