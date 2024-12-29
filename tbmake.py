"""
  File    : tbmake.py
  Purpose : TinyBreaker Maker
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : Jan 1, 2025
  Repo    : https://github.com/martin-rizzo/TBMake
  License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                              TinyBreaker Maker
#   A command-line tool for creating TinyBreaker models by fusing PixArt with SD
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
import struct
import argparse
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):
    # modules that are not available in the standard library are imported here
    import numpy
    from safetensors       import safe_open
    from safetensors.numpy import save_file as save_safetensors


FSTAGE_TXENCODER_SD_KEY   = "first_stage_txmodel/encoder/sd15"
FSTAGE_TXDECODER_SD_KEY   = "first_stage_txmodel/decoder/sd15"
FSTAGE_TXENCODER_SDXL_KEY = "first_stage_txmodel/encoder/sdxl"
FSTAGE_TXDECODER_SDXL_KEY = "first_stage_txmodel/decoder/sdxl"
FSTAGE_HQENCODER_SD_KEY   = "first_stage_hqmodel/encoder/sd15"
FSTAGE_HQDECODER_SD_KEY   = "first_stage_hqmodel/decoder/sd15"
FSTAGE_HQENCODER_SDXL_KEY = "first_stage_hqmodel/encoder/sdxl"
FSTAGE_HQDECODER_SDXL_KEY = "first_stage_hqmodel/decoder/sdxl"
BASE_MODEL_KEY            = "base_model"
BASE_COND_KEY             = "base_conditioner"
TRANSCODER_KEY            = "transcoder"
REFINER_MODEL_KEY         = "refiner_model"
REFINER_COND_KEY          = "refiner_conditioner"


# suffixes for each recognized model type
_SUFFIXES = {
    "decoder"          : "decoder.conv_in.weight",
    "encoder_diffusers": "decoder.up_blocks.0.resnets.0.norm1.weight",
    "encoder"          : "encoder.conv_in.weight",
    "encoder_diffusers": "encoder.down_blocks.0.resnets.1.norm2.weight",
}


# ANSI escape codes for colored terminal output
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
DEFAULT_COLOR = '\033[0m'

#----------------------------- ERROR MESSAGES ------------------------------#

def disable_colors():
    global RED, GREEN, YELLOW, CYAN, DEFAULT_COLOR
    RED, GREEN, YELLOW, CYAN, DEFAULT_COLOR = '', '', '', '', ''

def warning(message: str, *info_messages: str) -> None:
    """Displays and logs a warning message to the standard error stream.
    """
    print()
    print(f"{CYAN}[{YELLOW}WARNING{CYAN}]{DEFAULT_COLOR} {message}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {YELLOW}{info_message}{DEFAULT_COLOR}", file=sys.stderr)

def error(message: str, *info_messages: str) -> None:
    """Displays and logs an error message to the standard error stream.
    """
    print()
    print(f"{CYAN}[{RED}ERROR{CYAN}]{DEFAULT_COLOR} {message}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {RED}{info_message}{DEFAULT_COLOR}", file=sys.stderr)

def fatal_error(message: str, *info_messages: str) -> None:
    """Displays and logs an fatal error to the standard error stream and exits.
    Args:
        message       : The fatal error message to display.
        *info_messages: Optional informational messages to display after the error.
    """
    error(message)
    for info_message in info_messages:
        print(f" {CYAN}\u24d8  {info_message}{DEFAULT_COLOR}", file=sys.stderr)
    print()
    exit(1)

#--------------------------------- HELPERS ---------------------------------#

def is_terminal_output() -> bool:
    """Check if the standard output is connected to a terminal."""
    return sys.stdout.isatty()


def print_submodel_loc(name: str, location: tuple) -> None:
    """Prints the location of a submodel that will be part of the Tiny Breaker model."""
    path, prefix = location if location else ("", "")
    filename_and_prefix = "---"
    if path:
        filename_and_prefix = f"{os.path.basename(path)} [{prefix}]"
    print(f"  {CYAN}+ {name:<24}:{DEFAULT_COLOR} {YELLOW}{filename_and_prefix}{DEFAULT_COLOR}")


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
    #except IOError:
    #    fatal_error(f"Error reading the file '{path}'.")


def find_tensor_prefix(state_dict    : dict,
                       suffix        : str,
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
            if (not_containing is not None) and (not_containing in key):
                continue
            return key[:-len(suffix)]

    # if no key matches the suffix, return an empty string
    return ""





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

        state_dict = StateDict()
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
            path_and_prefix = ("", "")
        return cls.from_file(path_and_prefix[0], path_and_prefix[1], subprefix1, subprefix2)


    def with_prefix(self,
                    prefix    : str
                    ) -> "StateDict":
        prefix = normalize_prefix(prefix)
        return StateDict( { prefix + k: v for k, v in self.items() } )


#---------------------------- SUBMODELS FINDER -----------------------------#

def find_pixart_submodels(file_path: str) -> dict:
    """Finds the necessary submodels in a PixArt model file."""
    header = load_safetensors_header(file_path)

    # each found submodel will be added to this list by its prefix
    prefixes = {}
    prefixes[BASE_MODEL_KEY] = \
        find_tensor_prefix(header, suffix="t_block.1.weight") or \
        find_tensor_prefix(header, suffix="adaln_single.emb.timestep_embedder.linear_1.bias")

    # convert the `prefixes` to a dictionary of (file_path, prefix) tuples
    locations = { key: (file_path, prefix) for key, prefix in prefixes.items() if prefix is not None }
    return locations


def find_sd_submodels(file_path: str) -> dict:
    """Finds the necessary submodels in a Stable Diffusion 1.5 model file."""
    header = load_safetensors_header(file_path)

    # each found submodel will be added to this list by its prefix
    prefixes = {}
    prefixes[FSTAGE_HQENCODER_SD_KEY] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["encoder"          ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["encoder_diffusers"])
    prefixes[FSTAGE_HQDECODER_SD_KEY] = \
        find_tensor_prefix(header, suffix=_SUFFIXES["decoder"          ]) or \
        find_tensor_prefix(header, suffix=_SUFFIXES["decoder_diffusers"])

    # convert the `prefixes` to a dictionary of (file_path, prefix) tuples
    locations = { key: (file_path, prefix) for key, prefix in prefixes.items() if prefix is not None }
    return locations


#------------------------------ TINY BREAKER -------------------------------#

def make_tiny_breaker_with_sd15(submodels_loc: dict) -> dict:
    """Creates a TinyBreaker model from a PixArt model (base) and a Stable Diffusion 1.5 model (refiner).
    Args:
        submodels_loc (dict): A dictionary of submodel -> (file_path, prefix) with
                              the location of the each submodel to be used.
                              The `file_path` is the path to the model file.
                              The `prefix` is the prefix of the tensors in the model
                              file that belong to the submodel.
    """
    # by default we will use different encoder and decoder for the VAE,
    # a SDXL encoder to provide SDXL latent images to the PixArt model, and
    # a SD1.5 decoder to decode the latent images from the refiner.
    FSTAGE_TXENCODER_KEY = FSTAGE_TXENCODER_SDXL_KEY
    FSTAGE_TXDECODER_KEY = FSTAGE_TXDECODER_SD_KEY
    FSTAGE_HQENCODER_KEY = FSTAGE_HQENCODER_SDXL_KEY
    FSTAGE_HQDECODER_KEY = FSTAGE_HQDECODER_SD_KEY

    # show information about any submodel that will be used to create the TinyBreaker model
    print_submodel_loc("first stage    /encoder", submodels_loc.get(FSTAGE_TXENCODER_KEY))
    print_submodel_loc("first stage    /decoder", submodels_loc.get(FSTAGE_TXDECODER_KEY))
    print_submodel_loc("first stage HQ /encoder", submodels_loc.get(FSTAGE_HQENCODER_KEY))
    print_submodel_loc("first stage HQ /decoder", submodels_loc.get(FSTAGE_HQDECODER_KEY))
    print_submodel_loc("base model"             , submodels_loc.get(BASE_MODEL_KEY      ))
    print_submodel_loc("transcoder"             , submodels_loc.get(TRANSCODER_KEY      ))
    print_submodel_loc("refiner model"          , submodels_loc.get(REFINER_MODEL_KEY   ))
    print_submodel_loc("refiner conditioner"    , submodels_loc.get(REFINER_COND_KEY    ))

    if not FSTAGE_TXENCODER_KEY in submodels_loc:
        fatal_error("Missing first stage encoder.", "Some model containing a Tiny SDXL VAE encoder must be provided.")
    if not FSTAGE_TXDECODER_KEY in submodels_loc:
        fatal_error("Missing first stage decoder.", "Some model containing a Tiny SD1.5 VAE decoder must be provided.")
    if not FSTAGE_HQENCODER_KEY in submodels_loc:
        fatal_error("Missing first stage HQ encoder.", "Some model containing a SDXL VAE encoder must be provided.")
    if not FSTAGE_HQDECODER_KEY in submodels_loc:
        fatal_error("Missing first stage HQ decoder.", "Some model containing a SD1.5 VAE decoder must be provided.")

    state_dict = { }

    # fstage_hq_encoder = StateDict.from_location(submodels_loc.get(FSTAGE_TXENCODER_KEY), "encoder", "quant_conv")
    # fstage_hq_decoder = StateDict.from_location(submodels_loc.get(FSTAGE_TXDECODER_KEY), "decoder", "post_quant_conv")
    # base_model        = StateDict.from_location(submodels_loc.get(BASE_MODEL_KEY       ))
    # transcoder_model  = StateDict.from_location(submodels_loc.get(TRANSCODER_KEY       ))
    # refiner_model     = StateDict.from_location(submodels_loc.get(REFINER_MODEL_KEY    ))
    # refiner_cond      = StateDict.from_location(submodels_loc.get(REFINER_COND_KEY     ))

    # fstage_hq_encoder = fstage_hq_encoder.with_prefix("first_stage_hqmodel")
    # fstage_hq_decoder = fstage_hq_decoder.with_prefix("first_stage_hqmodel")
    # base_model        = base_model.with_prefix("base_model")
    # transcoder_model  = transcoder_model.with_prefix("transcoder")
    # refiner_model     = refiner_model.with_prefix("refiner_model")
    # refiner_cond      = refiner_cond.with_prefix("refiner_conditioner")

    # state_dict = {**fstage_hq_encoder, **fstage_hq_decoder}

    return state_dict


def make_tiny_breaker_with_sdxl(submodels_loc: dict) -> dict:
    """Creates a TinyBreaker model from a PixArt model (base) and a SDXL model (refiner)."""
    fatal_error("make_tiny_breaker_with_sdxl() is not implemented yet.",
                "Only SD1.5 refiners are supported at the moment, use the --sd option.")
    return {}


def make_tiny_breaker(submodels_loc: dict, refiner_type: str) -> dict:
    """
    Creates a TinyBreaker model from the given submodels.
    Args:
        submodels_loc (dict): A dictionary of submodel -> (file_path, prefix) with
                              the location of the each submodel to be used.
                              The `file_path` is the path to the model file.
                              The `prefix` is the prefix of the tensors in the model
                              file that belong to the submodel.
        refiner_type   (str): The type of refiner model to use for the TinyBreaker model.
                              Supported values: "sd15", "sdxl".
    """
    if refiner_type == "sd15":
        return make_tiny_breaker_with_sd15(submodels_loc)
    elif refiner_type == "sdxl":
        return make_tiny_breaker_with_sdxl(submodels_loc)
    else:
        fatal_error(f"Unknown model kind '{refiner_type}'",
                     "Only 'sd15' and 'sdxl' are supported")



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
    parser.add_argument(      "--pixart", help="Path to the PixArt model file used as base model.", nargs='*')
    parser.add_argument(      "--sd"    , help="Path to the SD model used as refiner.", nargs='*')
    parser.add_argument(      "--sdxl"  , help="Path to the SDXL model used as refiner.", nargs='*')
    parser.add_argument("-a", "--aux"   , help="Path to the auxiliary models", nargs='*')
    parser.add_argument("-r", "--res"   , help="The resolution for which the PixArt model was trained.", type=int, default=None)
    parser.add_argument("-c", "--color" , help="Use color output when connected to a terminal", action='store_true')
    parser.add_argument("--color-always", help="Always use color output", action='store_true')
    args = parser.parse_args(args)

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
    if not args.aux:
        fatal_error("You must specify the auxiliary model file (--aux).",
                    "An auxiliary model file is provided by default with the TinyBreaker project.")
    if len(args.aux) > 1:
        fatal_error("You can't specify more than one auxiliary model file (--aux).")
    if not args.res:
        fatal_error("You must specify the resolution for which the PixArt model was trained (--res).",
                    "This value is intrinsic to the PixArt model and determines the approximate size of the output image.",
                    "Typical values are 2048, 1024 or 512.")

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

    state_dict = make_tiny_breaker(submodels_loc, refiner_type = "sdxl" if args.sdxl else "sd15" )



if __name__ == "__main__":
    main()