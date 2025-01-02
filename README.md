<div align="center">

# Tiny Breaker Tools

<p>
<img alt="Platform" src="https://img.shields.io/badge/platform-Linux-33F">
<img alt="License"  src="https://img.shields.io/github/license/martin-rizzo/TinyBreakerTools?color=11D">
<img alt="Version"  src="https://img.shields.io/github/v/tag/martin-rizzo/TinyBreakerTools?label=version">
<img alt="Last"     src="https://img.shields.io/github/last-commit/martin-rizzo/TinyBreakerTools?color=33F">
</p>

<!-- Image -->
![Tiny Breaker Tools](./console.png)
</div>

**Tiny Breaker Tools** is a collection of command-line scripts designed for the creation and manipulation of "Tiny Breaker" models.

The following command is currently available:

- `maketb`: A command-line tool for creating custom Tiny Breaker models.

   *More tools are under development.*


## Tiny Breaker Model Overview

Tiny Breaker models are constructed by merging a pre-trained PixArt model with a Stable Diffusion (SD) refiner model. This strategic combination leverages the efficient generative capabilities of PixArt with the refinement power of SD models, resulting in high-quality image generation.


## Key Features

*   **Intuitive Command-Line Interface:**  The tools are designed for ease of use with clear and concise command-line arguments.
*   **Support for Multiple PixArt Resolutions:** Allows integration of PixArt models with various resolutions, including 512px, 1024px, and 2048px.
*   **Flexible Refiner Model Selection:** Supports both Stable Diffusion 1.5 (SD1.5) and Stable Diffusion XL (SDXL) refiner models.


## Installation

Currently, `maketb` is a command-line tool and does not require installation via pip. Simply clone the repository and ensure you have the required dependencies (e.g., PyTorch, Transformers, etc.) installed.

```bash
  git clone https://github.com/martin-rizzo/TinyBreakerTool
  cd TinyBreakerTool
  # create the python virtual environment with the dependencies of the project
  ./maketb.sh --create-venv
```


## Command "maketb"

The `maketb` command takes several arguments to create a TinyBreaker model. Here's how to use it:

### SD1.5 as Refiner

```maketb --pixart <pixart-model> --sd <sd_model> --resolution <pixart-resolution>```

-   `--pixart <pixart-model>`: Specifies the path to the PixArt model in safetensors format.
-   `--sd <sd_model>`: Specifies the path to the SD1.5 refiner model in safetensors format.
-   `--resolution <resolution>`: Declares the resolution for which the PixArt model was trained (e.g., 1024).

### SDXL as Refiner

```maketb --pixart <pixart-model>  --sdxl <sdxl_model> --resolution <pixart-resolution>```

-   `--pixart <pixart-model>`: Specifies the path to the PixArt model in safetensors format.
-   `--sdxl <sdxl_model>`: Specifies the path to the SDXL refiner model in safetensors format.
-   `--resolution <resolution>`: Declares the resolution for which the PixArt model was trained (e.g., 1024).

### Example

```bash
./maketb.sh --pixart pixart-alpha-base.safetensors \
       --sdxl sdxl-base-1.0.safetensors \
       --aux auxiliary.safetensors \
       --resolution 1024
```

This command will create a TinyBreaker model using `pixart-alpha-base.safetensors` as the PixArt main model, `sdxl-base-1.0.safetensors` as the SDXL refiner, and the provided `auxiliary.safetensors` as the auxiliary models, declaring a resolution of 1024px for the PixArt model.

To know more parameters, use `./maketb.sh --help`.


## Support Models

The `auxiliary.safetensors` file is crucial for the proper functioning of TinyBreaker models. This file contains pre-trained support models that are necessary for the merging process. **This file is already included in the repository.**


## License

Copyright (c) 2025 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.


