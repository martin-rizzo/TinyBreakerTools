"""
  File    : maketb.py
  Purpose : Creates TinyBreaker models by fusing PixArt with SD
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : Jan 1, 2025
  Repo    : https://github.com/martin-rizzo/TinyBreakerTool
  License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                              Tiny Breaker Tool
#       A set of scripts for creating and handling Tiny Breaker models.
#
#     Copyright (c) 2025 Martin Rizzo
#
#     Permission is hereby granted, free of charge, to any person obtaining
#     a copy of this software and associated documentation files (the
#     "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so, subject to
#     the following conditions:
#
#     The above copyright notice and this permission notice shall be
#     included in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import sys
import json
import base64
import struct
import argparse
from datetime import datetime
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):
    # modules that are not available in the standard library are imported here
    import numpy as np
    from tqdm              import tqdm
    from safetensors       import safe_open
    from safetensors.numpy import save as safetensors_bytes

# directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_AUXILIARY_MODEL_PATH = os.path.join(_SCRIPT_DIR, "auxiliary.safetensors")

# IDs of the different model types used by TinyBreaker
_FSTAGE_VAE_SD_ENCODER  = "first_stage_hqmodel (sd,encoder)"
_FSTAGE_VAE_SD_DECODER  = "first_stage_hqmodel (sd,decoder)"
_FSTAGE_VAE_XL_ENCODER  = "first_stage_hqmodel (xl,encoder)"
_FSTAGE_VAE_XL_DECODER  = "first_stage_hqmodel (xl,decoder)"
_FSTAGE_TINY_SD_ENCODER = "first_stage_model (sd,encoder)"
_FSTAGE_TINY_XL_ENCODER = "first_stage_model (xl,encoder)"
_FSTAGE_TINY_SD_DECODER = "first_stage_model (sd,decoder)"
_FSTAGE_TINY_XL_DECODER = "first_stage_model (xl,decoder)"
_BASE_MODEL             = "base.model"
_BASE_COND              = "base.conditioner"
_TRANSCODER_XL_SD       = "transcoder"
_REFINER_MODEL_SD       = "refiner.model (sd)"
_REFINER_MODEL_XL       = "refiner.model (xl)"
_REFINER_COND_SD        = "refiner.conditioner (sd)"
_REFINER_COND_XL        = "refiner.conditioner (xl)"

# suffixes for each recognized model type
_SUFFIXES = {
    "vae_decoder"          : "decoder.conv_in.weight",
    "vae_decoder_diffusers": "decoder.up_blocks.0.resnets.0.norm1.weight",
    "vae_encoder"          : "encoder.conv_in.weight",
    "vae_encoder_diffusers": "encoder.down_blocks.0.resnets.1.norm2.weight",
    "pixart"               : "t_block.1.weight",
    "pixart_diffusers"     : "adaln_single.emb.timestep_embedder.linear_1.bias",
    "sd15model"            : "input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight",
    "sd15model_diffusers"  : "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight",
    "sd15cond"             : "text_model.encoder.layers.8.self_attn.out_proj.weight",
    "sdxlmodel"            : "input_blocks.8.1.transformer_blocks.8.attn1.to_k.weight",
    "sdxlmodel_diffusers"  : "down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_k.weight",
    "sdxlcond"             : "embedders.1.model.transformer.resblocks.28.mlp.c_proj.weight",
}


# ANSI escape codes for colored terminal output
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
DKGRAY = '\033[90m'
RESET  = '\033[0m'

#----------------------------- ERROR MESSAGES ------------------------------#

def disable_colors():
    global RED, GREEN, YELLOW, CYAN, DKGRAY, RESET
    RED, GREEN, YELLOW, CYAN, DKGRAY, RESET = "", "", "", "", "", ""

def warning(message: str, *info_messages: str) -> None:
    """Displays and logs a warning message to the standard error stream.
    """
    print()
    print(f"{CYAN}[{YELLOW}WARNING{CYAN}]{RESET} {message}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {YELLOW}{info_message}{RESET}", file=sys.stderr)

def error(message: str, *info_messages: str) -> None:
    """Displays and logs an error message to the standard error stream.
    """
    print()
    print(f"{CYAN}[{RED}ERROR{CYAN}]{RESET} {message}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {RED}{info_message}{RESET}", file=sys.stderr)

def fatal_error(message: str, *info_messages: str) -> None:
    """Displays and logs an fatal error to the standard error stream and exits.
    Args:
        message       : The fatal error message to display.
        *info_messages: Optional informational messages to display after the error.
    """
    error(message)
    for info_message in info_messages:
        print(f" {CYAN}\u24d8  {info_message}{RESET}", file=sys.stderr)
    print()
    exit(1)

#--------------------------------- HELPERS ---------------------------------#

def is_terminal_output() -> bool:
    """Check if the standard output is connected to a terminal."""
    return sys.stdout.isatty()

def get_file_extension(path: str) -> str:
    """Returns the file extension from a given file path."""
    return os.path.splitext(path)[1]

def find_unique_path(path: str) -> str:
    """Returns the first available path to not overwrite an existing file."""
    if not os.path.exists(path):
        return path
    base_name, extension = os.path.splitext(path)
    for number in range(1, 1000000):
        new_path = f"{base_name}_{number:02d}{extension}"
        if not os.path.exists(new_path) or number == 999999:
            return new_path

def print_submodel_loc(name: str, location: tuple, no_location_message: str = None) -> None:
    """Prints the location of a submodel that will be part of the Tiny Breaker model."""
    path, prefix        = location if location else ("", "")
    filename            = os.path.basename(path).removesuffix(".safetensors")
    no_location_message = f"{GREEN}{no_location_message}" if no_location_message else f"{RED}missing!"
    if path:
        filename_and_prefix = f"{YELLOW}{filename:<26}  {DKGRAY}/{prefix}*"
    else:
        filename_and_prefix = no_location_message
    print(f"  {CYAN}+ {name:<26}:{RESET} {filename_and_prefix}{RESET}")


def normalize_prefix(prefix: str) -> str:
    """Normalizes any prefix to ensure it ends with a dot."""
    prefix = prefix.strip() if prefix else ""
    if prefix and not prefix.endswith('.'):
        prefix += '.'
    return prefix


def load_safetensors_header(path: str, prefix="") -> dict:
    """Returns a dictionary containing the header information of a safetensors file.
    Args:
        path   : The path to the safetensors file.
        prefix : An optional prefix to filter the returned tensor names.
                 (only tensors whose names start with this prefix will be included,
                  and the prefix will be removed from the keys in the returned dictionary)
    """
    extension = os.path.splitext(path)[1].lower()
    if extension != ".safetensors":
        fatal_error(f"Unsupported file format for '{os.path.basename(path)}'",
                     "The command expects a '.safetensors' file.")
    try:
        # load the safetensors header from the file
        with open(path, "rb") as f:
            header_length = struct.unpack('<Q', f.read(8))[0]
            header_data = f.read(header_length)
            header      = json.loads(header_data)

        # if not prefix is specified, return all tensors in the header
        if not prefix:
            return header

        # extract the tensors that match the prefix
        selected_tensors = { }
        prefix       = prefix.rstrip('.') + '.' if prefix else ""
        prefix_len   = len(prefix)
        for tensor_name, tensor_info in header.items():
            if tensor_name.startswith(prefix):
                key = tensor_name[prefix_len:]
                selected_tensors[ key ] = tensor_info
        return selected_tensors

    except json.JSONDecodeError as e:
        fatal_error(f"The file '{path}' does not have a valid JSON header: <{e}>")
    except IOError:
       fatal_error(f"Error reading the file '{path}'.")


def find_tensor_prefix(state_dict    : dict,
                       suffix        : str,
                       containing    : str = None,
                       not_containing: str = None
                       ) -> str:
    """
    Returns the prefix of a key in the state dictionary that matches the given suffix.
    Args:
        state_dict    (dict): The model parameters as a dictionary.
        suffix         (str): The suffix to match at the end of the key.
        not_containing (str): If provided, specifies that the keys returned should not contain this substring.
    """
    # iterate over all keys in the state dictionary
    for key in state_dict.keys():
        if key.endswith(suffix):
            if (containing is not None) and (containing not in key):
                continue
            if (not_containing is not None) and (not_containing in key):
                continue
            return key[:-len(suffix)]

    # if no key matches the suffix, return an empty string
    return ""


def load_kv_file(path: str) -> dict[str, str]:
    """
    Loads a file with key-value pairs and returns a dictionary containing them.

    The file has a key-value format similar to .ini files. e.g.: "key = value",
    but it supports ':' and corrects commas and semicolons at the end of the line.
    Returns:
        A dictionary containing the key-value pairs from the file.
    """
    try:
        with open(path, 'r') as file:
            content = file.read()
    except Exception as e:
        fatal_error(f"Error reading the file '{path}': {e}")

    output = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if   line.endswith(','): line = line[:-1]
        elif line.endswith(';'): line = line[:-1]
        if   ':' in line: separator = ':'
        elif '=' in line: separator = '='
        else:
            fatal_error(f"Invalid file format: '{line}' is not a valid key-value pair.")

        # split the line into a key-value pair
        key, value = line.split(separator, 1)
        key, value = key.strip(), value.strip()

        # remove quotes from string values
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        output[key] = str(value)

    return output


#---------------------------- STATE DICT CLASS -----------------------------#

class StateDict(dict):

    @classmethod
    def from_file(cls,
                  path      : str,
                  prefix    : str,
                  subprefix1: str = None,
                  subprefix2: str = None
                  ) -> "StateDict":
        """
        Returns a StateDict with tensors loaded from the specified file and prefix.
        Args:
            path       (str): The path to the file containing the tensors to load.
            prefix     (str): The prefix that defines the tensor keys to be loaded.
            subprefix1 (str): This string is added to prefix to define the tensor keys to be loaded.
            subprefix2 (str): This string is added to prefix to define the tensor keys to be loaded.

        Only the tensors with keys starting with the provided prefix will be loaded.
        But if any of the subprefixes is provided, the keys will be loaded if they
        start with the prefix+subprefix string.

        The prefix is always removed from the keys in the returned StateDict.
        Subprefixes are not removed.
        """

        # normalize prefixes
        prefix     = normalize_prefix(prefix)
        subprefix1 = normalize_prefix(subprefix1)
        subprefix2 = normalize_prefix(subprefix2)

        # prepare filter prefixes
        filter_prefix1 = prefix + (subprefix1 or "")
        filter_prefix2 = prefix + (subprefix2 or "")

        state_dict = cls()
        with safe_open(path, framework="numpy", device="cpu") as f:
            for key in f.keys():
                if key.startswith(filter_prefix1) or key.startswith(filter_prefix2):
                    new_key = key[len(prefix):]
                    state_dict[new_key] = f.get_tensor(key)

        return state_dict



    @classmethod
    def from_location(cls,
                      path_and_prefix: tuple,
                      subprefix1     : str = None,
                      subprefix2     : str = None
                      ) -> "StateDict":
        """
        Returns a StateDict with tensors loaded from the specified location (path and prefix)
        Args:
            path_and_prefix (tuple): A tuple containing the path to the file and the prefix for tensor keys.
            subprefix1       (str) : This string is added to prefix to define the tensor keys to be loaded.
            subprefix2       (str) : This string is added to prefix to define the tensor keys to be loaded.

        Only the tensors with keys starting with the provided prefix will be loaded.
        But if any of the subprefixes is provided, the keys will be loaded if they
        start with the prefix+subprefix string.

        The prefix is always removed from the keys in the returned StateDict.
        Subprefixes are not removed.
        """
        if not isinstance(path_and_prefix, tuple) or len(path_and_prefix) != 2:
            return cls()
        return cls.from_file(path_and_prefix[0], path_and_prefix[1], subprefix1, subprefix2)


    @classmethod
    def from_submodels(cls, submodel_descriptions: list[tuple]) -> "StateDict":
        """
        Creates a StateDict by combining the state dictionaries from multiple submodels.
        Args:
            submodel_descriptions (list[tuple]): A list of tuples, where each tuple contains:
                - A prefix string: This prefix is added to the keys of the loaded state dictionary.
                - A tuple: Arguments to be passed to the `from_location` class method,
                           specifying the path and prefix for loading the submodel's state dictionary.
        Returns:
            StateDict: A StateDict containing the combined state dictionaries from all submodels.
                       The keys in the returned StateDict will be prefixed with the provided prefix.
                       Each submodel's state dictionary is converted to float16.
        """
        state_dict = cls()
        for prefix, from_location_args in tqdm(submodel_descriptions, desc="Loading submodels ", unit="model"):
            submodel = StateDict.from_location(*from_location_args).with_prefix(prefix).to(np.float16)
            state_dict.update(submodel)
        return state_dict


    def save_as_safetensors(self,
                            path     : str,
                            *,# keyword-only arguments #
                            metadata : dict[str, str] = None,
                            overwrite: bool           = False,
                            ) -> None:
        """
        Saves the tensors in this StateDict as a safetensors file at 'path'.
        Args:
            path      (str) : The file path to save the safetensors file.
            metadata  (dict): A dictionary containing additional metadata for the safetensors file.
            overwrite (bool): If True, overwrites the existing file at 'path'.
        """
        if not get_file_extension(path):
            path += ".safetensors"
        if not overwrite:
            path = find_unique_path(path)

        # serialize the tensor dictionary and metadata into bytes using safetensors library
        byte_data = safetensors_bytes(self, metadata=metadata )

        # write the binary data to the file displaying a progress bar
        chunk_size = 1024 * 1024
        with open(path, "wb") as f:
            with tqdm(total=len(byte_data), unit="B", unit_scale=True, desc="Saving safetensors") as pbar:
                for i in range(0, len(byte_data), chunk_size):
                    f.write(byte_data[i:i+chunk_size])
                    pbar.update( min(len(byte_data)-i, chunk_size) )


    def to(self, dtype) -> "StateDict":
        """
        Casts all tensors in the StateDict to the specified dtype, modifying the StateDict in-place.
        Args:
            dtype (np.dtype): The desired dtype for the tensors.
        Returns:
            Returns self for chaining.
        """
        for key, tensor in self.items():
            self[key] = tensor.astype(dtype)
        return self


    def with_prefix(self, prefix: str) -> "StateDict":
        """Returns a new StateDict with all keys prefixed by 'prefix'."""
        prefix = normalize_prefix(prefix)
        return StateDict( { prefix + k: v for k, v in self.items() } )


#---------------------------- SUBMODEL FINDERS -----------------------------#

def find_pixart_submodels(file_path: str) -> dict:
    """Finds the necessary submodels in a PixArt model file. (--pixart)"""
    header = load_safetensors_header(file_path)

    # each found submodel will be added to this list by its prefix
    prefixes = {}
    prefixes[_BASE_MODEL] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["pixart"          ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["pixart_diffusers"])

    # convert the `prefixes` to a dictionary of (file_path, prefix) tuples
    locations = { key: (file_path, prefix) for key, prefix in prefixes.items() if prefix is not None }
    return locations


def find_sd_submodels(file_path: str) -> dict:
    """Finds the necessary submodels in a Stable Diffusion 1.5 model file. (--sd)"""
    header = load_safetensors_header(file_path)

    # each found submodel will be added to this list by its prefix
    prefixes = {}
    prefixes[_FSTAGE_VAE_SD_ENCODER] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_encoder"          ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_encoder_diffusers"])
    prefixes[_FSTAGE_VAE_SD_DECODER] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_decoder"          ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_decoder_diffusers"])
    prefixes[_REFINER_COND_SD] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["sd15cond"             ])
    prefixes[_REFINER_MODEL_SD] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["sd15model"            ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["sd15model_diffusers"  ])

    # convert the `prefixes` to a dictionary of (file_path, prefix) tuples
    locations = { key: (file_path, prefix) for key, prefix in prefixes.items() if prefix is not None }
    return locations


def find_sdxl_submodels(file_path: str) -> dict:
    """Finds the necessary submodels in a Stable Diffusion XL model file. (--sdxl)"""
    header = load_safetensors_header(file_path)

    # each found submodel will be added to this list by its prefix
    prefixes = {}
    prefixes[_FSTAGE_VAE_XL_ENCODER] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_encoder"          ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_encoder_diffusers"])
    prefixes[_FSTAGE_VAE_XL_DECODER] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_decoder"          ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["vae_decoder_diffusers"])
    prefixes[_REFINER_COND_XL] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["sdxlcond"             ])
    prefixes[_REFINER_MODEL_XL] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["sdxlmodel"            ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["sdxlmodel_diffusers"  ])

    # convert the `prefixes` to a dictionary of (file_path, prefix) tuples
    locations = { key: (file_path, prefix) for key, prefix in prefixes.items() if prefix is not None }
    return locations


def find_auxiliary_submodels(file_path: str) -> dict:
    """Finds the necessary auxiliary models used by TinyBreaker. (--aux)"""
    header = load_safetensors_header(file_path)

    # each found submodel will be added to this list by its prefix
    prefixes = {}
    prefixes[_FSTAGE_TINY_SD_ENCODER] = find_tensor_prefix(header, suffix="encoder.3.conv.4.weight", containing="sd")
    prefixes[_FSTAGE_TINY_SD_DECODER] = find_tensor_prefix(header, suffix="decoder.3.conv.4.weight", containing="sd")
    prefixes[_FSTAGE_TINY_XL_ENCODER] = find_tensor_prefix(header, suffix="encoder.3.conv.4.weight", containing="xl")
    prefixes[_FSTAGE_TINY_XL_DECODER] = find_tensor_prefix(header, suffix="decoder.3.conv.4.weight", containing="xl")
    prefixes[_TRANSCODER_XL_SD      ] = find_tensor_prefix(header, suffix="transe.3.conv.4.weight" , containing="transcoder")

    # convert the `prefixes` to a dictionary of (file_path, prefix) tuples
    locations = { key: (file_path, prefix) for key, prefix in prefixes.items() if prefix is not None }
    return locations

#-------------------------------- METADATA ---------------------------------#

def read_thumbnail_in_base64(path: str) -> str:
    """Returns the thumbnail image from the given path as a base64 encoded string."""
    try:
        # check if the file exists
        if not os.path.exists(path):
            fatal_error(f"File '{path}' does not exist.")

        # check if the file size exceeds 30 KB
        file_size_kb = os.path.getsize(path) / 1024
        if file_size_kb > 30:
            fatal_error("Thumbnail is too large, it will not be loaded.",
                        "It is recommended to use an image smaller than 30 KB with a resolution of 256x256.")

        # read the binary data from the file and convert to base64
        with open(path, 'rb') as file:
            binary_data = file.read()
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        return base64_data

    except IOError:
        fatal_error(f"Error reading the file '{path}'.")


def make_metadata(resolution: int,
                  *,# keyword-only arguments #
                  title         : str = None,
                  description   : str = None,
                  author        : str = None,
                  license       : str = None,
                  thumbnail_path: str = None,
                  ) -> dict[str, str]:
    """
    Creates a dictionary containing metadata for the model.
    Args:
        resolution    : The resolution of the model.
        title         : The title of the model.
        description   : The description of the model.
        author        : The author of the model.
        license       : The license under which the model is distributed.
        thumbnail_path: The path to the thumbnail image.
    Returns:
        A dictionary containing the metadata for the model.
    """
    # check if thumbnail is a jpg file
    thumbnail_ext = os.path.splitext(thumbnail_path)[1].lower()
    if thumbnail_ext not in [".jpg", ".jpeg"]:
        fatal_error(f"Thumbnail must be a jpg file, but '{thumbnail_path}' is a {thumbnail_ext} file.")

    # la fecha debe estar en isoformat pero solo la fecha, no la hora ni nada mas

    iso_8601_date    = datetime.now().strftime("%Y-%m-%d")
    thumbnail_base64 = read_thumbnail_in_base64(thumbnail_path) if thumbnail_path else None
    title            = title       or "TinyBreaker model"
    description      = description or "TinyBreaker is a hybrid model, a convergence of the poetic energy of PixArt and the explosive depth of SD1."

    # fill the required fields
    metadata = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture"  : "TinyBreaker",
        "modelspec.implementation": "reference",
        "modelspec.title"         : title,
        "modelspec.resolution"    : f"{resolution}x{resolution}",
        "modelspec.description"   : description,
        "modelspec.date"          : iso_8601_date,
    }
    # fill the optional fields
    if author:
        metadata["modelspec.author"] = author
    if license:
        metadata["modelspec.license"] = license
    if thumbnail_base64:
        metadata["modelspec.thumbnail"] = "data:image/jpeg;base64," + thumbnail_base64

    return metadata


#========================== TINY BREAKER CREATION ==========================#


def make_prototype0_sd15(submodels_loc: dict) -> StateDict:
    """Creates a TinyBreaker model from a PixArt model (base) and a Stable Diffusion 1.5 model (refiner).
    Args:
        submodels_loc: A dictionary of "submodel" -> (file_path, prefix) with the
                       location of the each submodel to be used.
                       The `file_path` is the path to the model file.
                       The `prefix` is the prefix of the tensors in the model file.
    """
    # VAE + HQ-VAE
    FSTAGE_MODEL_ENC   = _FSTAGE_TINY_XL_ENCODER  # <- Tiny SDXL   (encoder)
    FSTAGE_MODEL_DEC   = _FSTAGE_TINY_SD_DECODER  # <- Tiny SD1.5  (decoder)
    FSTAGE_HQMODEL_ENC = "discarded"              # None
    FSTAGE_HQMODEL_DEC = _FSTAGE_VAE_SD_DECODER   # <- SD1.5 (decoder)
    # BASE
    BASE_MODEL = _BASE_MODEL  # <- PixArt
    BASE_COND  = _BASE_COND   # <- T5 Encoder
    # REFINER
    REFINER_MODEL = _REFINER_MODEL_SD  # <- SD1.5
    REFINER_COND  = _REFINER_COND_SD   # <- SD1.5

    # show information about each submodel used to create TinyBreaker
    print()
    print(f"{CYAN}TinyBreaker prototype0 submodels{RESET}")
    print_submodel_loc("first stage HQ .encoder"  , submodels_loc.get(FSTAGE_HQMODEL_ENC), ">> discarded")
    print_submodel_loc("first stage HQ .decoder"  , submodels_loc.get(FSTAGE_HQMODEL_DEC))
    print_submodel_loc("first stage    .encoder"  , submodels_loc.get(FSTAGE_MODEL_ENC  ))
    print_submodel_loc("first stage    .decoder"  , submodels_loc.get(FSTAGE_MODEL_DEC  ))
    print_submodel_loc("base model"               , submodels_loc.get(BASE_MODEL        ))
    print_submodel_loc("base conditioner"         , submodels_loc.get(BASE_COND         ), ">> external t5-encoder")

    print_submodel_loc("transcoder", submodels_loc.get(_TRANSCODER_XL_SD ))

    print_submodel_loc("refiner model"      , submodels_loc.get(REFINER_MODEL))
    print_submodel_loc("refiner conditioner", submodels_loc.get(REFINER_COND ))
    print()

    if not _FSTAGE_TINY_XL_ENCODER in submodels_loc:
        fatal_error("Missing first stage encoder.", "Some model containing a Tiny SDXL VAE encoder must be provided.")
    if not _FSTAGE_TINY_SD_DECODER in submodels_loc:
        fatal_error("Missing first stage decoder.", "Some model containing a Tiny SD1.5 VAE decoder must be provided.")
    if not _FSTAGE_VAE_SD_DECODER in submodels_loc:
        fatal_error("Missing first stage HQ decoder.", "Some model containing a SD1.5 VAE decoder must be provided.")

    tinybreaker_submodels = [
      #                     StateDict.from_location(path_and_prefix, subprefix1, subprefix2)          |
      #--------------------------+------------------------------------+------------+------------------+
      # prefix                   |          path_and_prefix           | subprefix1 | subprefix2       |
      ("first_stage_hqmodel"    ,(submodels_loc.get(FSTAGE_HQMODEL_DEC), "decoder", "post_quant_conv")),
      ("first_stage_model"      ,(submodels_loc.get(FSTAGE_MODEL_ENC  ), "encoder"                   )),
      ("first_stage_model"      ,(submodels_loc.get(FSTAGE_MODEL_DEC  ), "decoder"                   )),
      ("base.diffusion_model"   ,(submodels_loc.get(BASE_MODEL        ),                             )),
      ("transcoder"             ,(submodels_loc.get(_TRANSCODER_XL_SD ),                             )),
      ("refiner.diffusion_model",(submodels_loc.get(REFINER_MODEL     ),                             )),
      ("refiner.conditioner"    ,(submodels_loc.get(REFINER_COND      ),                             )),
    ]
    return StateDict.from_submodels(tinybreaker_submodels)



def make_prototype1_sd15(submodels_loc: dict) -> StateDict:
    """Creates the TinyBreaker model (prototype1) with Stable Diffusion 1.5 as refiner.
    Args:
        submodels_loc: A dictionary of "submodel" -> (file_path, prefix) with the
                       location of the each submodel to be used.
                       The `file_path` is the path to the model file.
                       The `prefix` is the prefix of the tensors in the model file.
    """
    # VAE
    FSTAGE_MODEL_ENC = _FSTAGE_TINY_XL_ENCODER  # <- Tiny SDXL (encoder)
    FSTAGE_MODEL_DEC = _FSTAGE_VAE_SD_DECODER   # <- SD1.5     (decoder)
    # BASE
    BASE_MODEL = _BASE_MODEL  # <- PixArt
    BASE_COND  = _BASE_COND   # <- T5 Encoder
    # REFINER
    REFINER_FSTAGE_MODEL_ENC = _FSTAGE_TINY_SD_ENCODER  # <- Tiny SD1.5 (encoder)
    REFINER_FSTAGE_MODEL_DEC = _FSTAGE_TINY_SD_DECODER  # <- Tiny SD1.5 (decoder)
    REFINER_MODEL            = _REFINER_MODEL_SD        # <- SD1.5
    REFINER_COND             = _REFINER_COND_SD         # <- SD1.5

    # show information about each submodel used to create TinyBreaker
    print()
    print(f"{CYAN}TinyBreaker prototype1 submodels{RESET}")
    print_submodel_loc("first stage .encoder", submodels_loc.get(FSTAGE_MODEL_ENC))
    print_submodel_loc("first stage .decoder", submodels_loc.get(FSTAGE_MODEL_DEC))
    print_submodel_loc("base model"          , submodels_loc.get(BASE_MODEL      ))
    print_submodel_loc("base conditioner"    , submodels_loc.get(BASE_COND       ), ">> external t5-encoder")

    print_submodel_loc("transcoder", submodels_loc.get(_TRANSCODER_XL_SD))

    print_submodel_loc("refiner 1st stage .encoder", submodels_loc.get(REFINER_FSTAGE_MODEL_ENC))
    print_submodel_loc("refiner 1st stage .decoder", submodels_loc.get(REFINER_FSTAGE_MODEL_DEC))
    print_submodel_loc("refiner model"             , submodels_loc.get(REFINER_MODEL           ))
    print_submodel_loc("refiner conditioner "      , submodels_loc.get(REFINER_COND            ))
    print()

    tinybreaker_submodels = [
      #                       StateDict.from_location(path_and_prefix, subprefix1, subprefix2)                 |
      #----------------------------+------------------------------------------+------------+-------------------+
      # prefix                     |           path_and_prefix                | subprefix1 | subprefix2        |
      ("first_stage_model"        ,(submodels_loc.get(FSTAGE_MODEL_ENC        ), "encoder"                    )),
      ("first_stage_model"        ,(submodels_loc.get(FSTAGE_MODEL_DEC        ), "decoder", "post_quant_conv" )),
      ("base.diffusion_model"     ,(submodels_loc.get(BASE_MODEL              ),                              )),
      ("transcoder"               ,(submodels_loc.get(_TRANSCODER_XL_SD       ),                              )),
      ("refiner.first_stage_model",(submodels_loc.get(REFINER_FSTAGE_MODEL_ENC), "encoder"                    )),
      ("refiner.first_stage_model",(submodels_loc.get(REFINER_FSTAGE_MODEL_DEC), "decoder"                    )),
      ("refiner.diffusion_model"  ,(submodels_loc.get(REFINER_MODEL           ),                              )),
      ("refiner.conditioner"      ,(submodels_loc.get(REFINER_COND            ),                              )),
    ]
    return StateDict.from_submodels(tinybreaker_submodels)



def make_prototype1_sdxl(submodels_loc: dict) -> StateDict:
    """Creates a TinyBreaker model (PROTOTYPE1) with SDXL as refiner.
    Args:
        submodels_loc: A dictionary of "submodel" -> (file_path, prefix) with the
                       location of the each submodel to be used.
                       The `file_path` is the path to the model file.
                       The `prefix` is the prefix of the tensors in the model file.
    """
    fatal_error("make_prototype1_sdxl() is not implemented yet.",
                "Only SD1.5 refiners are supported at the moment, use the --sd option.")
    return StateDict()



def make_tinybreaker(submodels_path_prefix: dict,
                     refiner_type         : str,
                     version              : str = "prototype1"
                     ) -> StateDict:
    """
    Creates a TinyBreaker model from the given submodels.
    Args:
        submodels_path_prefix: A dictionary of submodel -> (file_path, prefix) with
                               the location of the each submodel to be used.
                               The `file_path` is the path to the model file.
                               The `prefix` is the prefix of the tensors in the model
                               file that belong to the submodel.
        refiner_type: The type of refiner model to use for the TinyBreaker model.
                      Supported values: "sd15", "sdxl".
        version     : The version of the TinyBreaker model to create.
                      Supported values: "prototype0", "prototype1".
    """
    if version == "prototype0":
        return make_prototype0_sd15(submodels_path_prefix)

    elif version == "prototype1":
        if refiner_type == "sd15":
            return make_prototype1_sd15(submodels_path_prefix)
        elif refiner_type == "sdxl":
            return make_prototype1_sdxl(submodels_path_prefix)
        else:
            fatal_error(f"Unknown refiner type '{refiner_type}'. Only 'sd15' and 'sdxl' are supported")

    else:
        fatal_error(f"Unknown model version '{version}'. Only 'prototype0' and 'prototype1' are supported")



#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args=None, parent_script=None):
    prog = None
    if parent_script:
        prog = parent_script + " " + os.path.basename(__file__).split('.')[0]

    parser = argparse.ArgumentParser(
        prog=prog,
        description="A command-line tool for creating TinyBreaker models by fusing PixArt with SD.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(      "--pixart"     , help="Path to the PixArt model file used as base model.", nargs='*')
    parser.add_argument(      "--sd"         , help="Path to the SD model used as refiner.", nargs='*')
    parser.add_argument(      "--sdxl"       , help="Path to the SDXL model used as refiner.", nargs='*')
    parser.add_argument(      "--aux"        , help="Path to the auxiliary models", nargs='*')
    parser.add_argument("-r", "--resolution" , help="The resolution for which the PixArt model was trained.", type=int, default=None)
    parser.add_argument("-t", "--title"      , help="The title of the model. e.g. 'TinyBreaker model'", type=str, default=None)
    parser.add_argument("-d", "--description", help="The description of the model. e.g. 'TinyBreaker model trained on 10k images'", type=str, default=None)
    parser.add_argument("-a", "--author"     , help="The author of the model.", type=str, default=None)
    parser.add_argument("-l", "--license"    , help="The license of the model. e.g. 'CC BY-NC-SA 4.0'", type=str, default=None)
    parser.add_argument("-m", "--metadata"   , help="A .ini file containing any extra metadata to include in the model.", type=str, default=None)
    parser.add_argument(      "--thumbnail"  , help="The path to the thumbnail image for the model.", type=str, default=None)
    parser.add_argument("-o", "--output"     , help="the output file name", type=str, default="output")
    parser.add_argument("-c", "--color"      , help="Use color output when connected to a terminal", action='store_true')
    parser.add_argument(      "--model-type" , help="The type of model to create. Supported values: 'prototype0', 'prototype1'.", type=str, default='prototype1')
    parser.add_argument("--color-always"     , help="Always use color output", action='store_true')
    args = parser.parse_args(args)

    args_output_file     = args.output
    args_auxiliary_paths = []

    # determine if color should be used
    use_color = args.color_always or (args.color and is_terminal_output())
    if not use_color:
        disable_colors()

    # check arguments
    if not args.pixart:
        fatal_error("You must specify a PixArt model (--pixart).")
    if not (args.sd or args.sdxl):
        fatal_error("You must specify either the SD model (--sd) or the SDXL model (--sdxl).")
    if args.sd and args.sdxl:
        fatal_error("You can't specify both the SD model (--sd) and the SDXL model (--sdxl).")
    if not args.resolution:
        fatal_error("You must specify the resolution for which the PixArt model was trained (--res).",
                    "This value is intrinsic to the PixArt model and determines the approximate size of the output image.",
                    "Typical values are 2048, 1024 or 512.")

    # generate a list of paths to auxiliary models in `auxiliary_paths`,
    # if the user did not provide any auxiliary model, then the default one will be used.
    if args.aux:
        args_auxiliary_paths = args.aux
    elif os.path.exists(_DEFAULT_AUXILIARY_MODEL_PATH):
        args_auxiliary_paths = [_DEFAULT_AUXILIARY_MODEL_PATH]


    submodels_loc = {}

    # collect the necessary submodels from all provided PixArt models
    for pixart_path in args.pixart:
        submodels_pixart = find_pixart_submodels(pixart_path)
        if not submodels_pixart:
            fatal_error(f"Cannot find a necessary tensor in '{os.path.basename(pixart_path)}'",
                         "Please ensure that you are using a valid PixArt model file.")
        submodels_loc.update(submodels_pixart)

    # collect the necessary submodels from all provided SD models
    for sd_path in args.sd or []:
        submodels_sd = find_sd_submodels(sd_path)
        if not submodels_sd:
            fatal_error(f"Cannot find a necessary tensor in '{os.path.basename(sd_path)}'",
                         "Please ensure that you are using a valid SD model file.")
        submodels_loc.update(submodels_sd)

    # collect the necessary submodels from all provided SDXL models
    for sdxl_path in args.sdxl or []:
        submodels_sdxl = find_sdxl_submodels(sdxl_path)
        if not submodels_sdxl:
            fatal_error(f"Cannot find a necessary tensor in '{os.path.basename(sdxl_path)}'",
                         "Please ensure that you are using a valid SDXL model file.")
        submodels_loc.update(submodels_sdxl)

    # collect the necessary submodels from all provided auxiliary models
    for aux_path in args_auxiliary_paths:
        submodels_aux = find_auxiliary_submodels(aux_path)
        if not submodels_aux:
            fatal_error(f"Cannot find a necessary tensor in '{os.path.basename(aux_path)}'",
                         "Please ensure that you are using a valid auxiliary model file.")
        submodels_loc.update(submodels_aux)

    # generate model and metadata
    state_dict = make_tinybreaker(submodels_loc,
                                  refiner_type = "sdxl" if args.sdxl else "sd15",
                                  version      = args.model_type,
                                  )
    metadata   = make_metadata(resolution     = args.resolution,
                               title          = args.title,
                               description    = args.description,
                               author         = args.author,
                               license        = args.license,
                               thumbnail_path = args.thumbnail,
                               )
    # try to load the extra metadata from the provided .ini file
    extra_metadata = load_kv_file(args.metadata) if args.metadata else None
    if extra_metadata:
        extra_metadata.update(metadata)
        metadata = extra_metadata

    # save model
    if state_dict:
        state_dict.save_as_safetensors(args_output_file,
                                       metadata  = metadata,
                                       overwrite = False)
    print()



if __name__ == "__main__":
    main()