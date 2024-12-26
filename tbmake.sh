#!/usr/bin/env bash
# File    : tbmake.sh
# Brief   : Wrapper for `tbmake.py` that automatically handles the python virtual env
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Jan 1, 2025
# Repo    : https://github.com/martin-rizzo/TBMake
# License : MIT
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
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)           # script name without extension
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")   # script directory
PYTHON_SCRIPT="${SCRIPT_DIR}/${SCRIPT_NAME}.py"           # name of the Python script to run
NON_ESSENTIAL_OPTIONS=( "-c" "--color" "--color-always" ) # options that do not trigger any action by themselves


# ANSI escape codes for colored terminal output
RED='\e[91m'
CYAN='\e[96m'
YELLOW='\e[93m'
DEFAULT_COLOR='\e[0m'

# Display a warning message
warning() {
    local message=$1
    echo
    echo -e "${CYAN}[${YELLOW}WARNING${CYAN}]${DEFAULT_COLOR} $message" >&2
}

# Display an error message
error() {
    local message=$1
    echo
    echo -e "${CYAN}[${RED}ERROR${CYAN}]${DEFAULT_COLOR} $message" >&2
}

# Displays a fatal error message and exits the script with status code 1
fatal_error() {
    local error_message=$1
    error "$error_message"
    shift
    # print informational messages, if any were provided
    while [[ $# -gt 0 ]]; do
        local info_message=$1
        echo -e " ${CYAN}\xF0\x9F\x9B\x88 $info_message${DEFAULT_COLOR}" >&2
        shift
    done
    echo
    exit 1
}

# Create and activate the python virtual environment
create_venv() {
    if [[ -d "venv" ]]; then
        echo "Virtual environment already exists."
        return
    fi
    echo "Creating virtual environment..."
    if ! python3 -m venv venv; then
        fatal_error "Virtual environment creation failed." \
                    "Please check if python3 and venv are installed on your system."
    fi
    echo "Virtual environment created."
}

# Activate the python virtual environment
activate_venv() {
    if [[ ! -f "venv/bin/activate" ]]; then
        fatal_error "The virtual environment does not exist." \
                    "you can use --create-venv to create it"
    fi
    # shellcheck disable=SC1091
    if ! source "venv/bin/activate"; then
        fatal_error "Error when activating virtual environment, it might be corrupted." \
                    "You can use --recreate-venv to recreate the virtual environment."
    fi
}

# Install dependencies from requirements.txt file if it exists
install_dependencies() {
    local requirements_file=$1
    if [[ ! -f "$requirements_file" ]]; then
        fatal_error "No '$requirements_file' file found." \
                    "Please check the project instalation instructions."
    fi
    if ! pip install --upgrade pip; then
        # failed to upgrade pip isn´t a fatal error, just a warning
        warning "Error when upgrading pip."
    fi
    if ! pip install -r "$requirements_file"; then
        fatal_error "Error when installing dependencies." \
                    "'pip' failed to install some packages, that might be due to network issues or incompatible packages."
    fi
    echo "Dependencies installed successfully."
}

# Check if a given option is non-essential
# (non-essential options do not trigger any action by themselves)
is_non_essential_option() {
    local option=$1
    [[ -z "$option" ]] && return 0
    for non_essential_option in "${NON_ESSENTIAL_OPTIONS[@]}"; do
        [[ "$option" == "$non_essential_option" ]] && return 0
    done
    return 1
}


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

# change to the directory where this script is located
cd "${SCRIPT_DIR}" \
 || fatal_error "Could not change to script directory." "${SCRIPT_DIR}" \
                "Explaining: This can be an erroneous path or a permission issue."

# verify if any extra options are passed as arguments
CREATE_VENV=false
REMOVE_VENV=false
SHOW_HELP=false

if [[ $# -le 1 ]] && is_non_essential_option "$1"; then
    # if no arguments are passed, the help message will be displayed
    SHOW_HELP=true
else
    # loop through the arguments and set the corresponding
    # variables to true if they match the options
    for arg in "$@"; do
        case $arg in
            -h | --help)
                SHOW_HELP=true
                ;;
            --create-venv)
                CREATE_VENV=true
                ;;
            --remove-venv)
                REMOVE_VENV=true
                ;;
            --recreate-venv)
                REMOVE_VENV=true
                CREATE_VENV=true
                ;;
        esac
    done
fi

# handle the help option
if [[ "$SHOW_HELP" == true ]]; then
    python3 "$PYTHON_SCRIPT" --help
    echo
    echo "wrapper options:"
    echo "  --create-venv      Create the python virtual environment"
    echo "  --remove-venv      Remove the python virtual environment"
    echo "  --recreate-venv    Remove and recreate the python virtual environment"
    echo
    exit 0
fi

# handle the extra options for removing the venv
if [[ "$REMOVE_VENV" == true ]]; then
    if [[ ! -d "venv" ]]; then
        fatal_error "No 'venv' directory found." \
            "You must create a virtual environment before removing it." \
            "Use the '--create-venv' option to create a new one."
    fi
    rm -rf venv
    echo "Virtual environment removed."
    exit 0
fi

# handle the extra options for creating the venv
if [[ "$CREATE_VENV" == true ]]; then
    create_venv
    activate_venv
    install_dependencies 'requirements.txt'
    exit 0
fi

# if no extra options are passed, just run the script normally
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    python_script_name=$(basename "$PYTHON_SCRIPT")
    fatal_error "Python script not found." \
                "Please ensure that the Python script '${python_script_name}' exists in the same directory as this bash wrapper."
fi
activate_venv
python3 "$PYTHON_SCRIPT" "$@"
