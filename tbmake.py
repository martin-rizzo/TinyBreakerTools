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
import argparse
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):
    # modules that are not available in the standard library are imported here
    import numpy
    import safetensors


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
    parser.add_argument("--pixart"      , help="Path to the PixArt model file used as base model.", nargs='*')
    parser.add_argument("--sd"          , help="Path to the SD model used as refiner.", nargs='*')
    parser.add_argument("--sdxl"        , help="Path to the SDXL model used as refiner.", nargs='*')
    parser.add_argument("--auxiliary"   , help="Path to the auxiliary models", nargs='*')
    parser.add_argument("--resolution"  , help="The resolution for which the PixArt model was trained.", type=int, default=None)
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
    if not args.auxiliary:
        fatal_error("You must specify the auxiliary model file (--auxiliary).",
                    "An auxiliary model file is provided by default with the TinyBreaker project.")
    if len(args.auxiliary) > 1:
        fatal_error("You can't specify more than one auxiliary model file (--auxiliary).")
    if not args.resolution:
        fatal_error("You must specify the resolution for which the PixArt model was trained (--resolution).",
                    "This value is intrinsic to the PixArt model and determines the approximate size of the output image.",
                    "Typical values are 2048, 1024 or 512.")


if __name__ == "__main__":
    main()