#!/bin/zsh

set -e

# Aim:
# Input:
# Output:
# Usage:
# Example:
# Dependencies:

##############################################################################
# FUNCTION                                                                   #
##############################################################################
usage() {
  echo "Usage: $(basename $0) --run_name|-n <run_name> --ckpt|-m <checkpoint> --config|-c <config> --outdir|-o <output_dir> [--level <log_level>] [--help]"
  echo "Options:"
  echo "  --run_name|-n : wandb run name"
  echo "  --ckpt|-m     : checkpoint"
  echo "  --config|-c   : config file"
  echo "  --outdir|-o   : output directory"
  echo "  --level       : log level (DEBUG, INFO, WARNING, ERROR)"
  echo "  --help        : print this help message"
  # add example
  echo "Example:"
  echo "$(basename $0) -n jumping-sweep-22 -m ./assets/jumping-sweep-22/rank_0.pt -c ./assets/jumping-sweep-22/config.yaml -o ./metrics"
  exit 1
}

# Function to print timestamp
print_timestamp() {
  date +"%Y%m%d-%H%M%S" # e.g. 20240318-085729
}

# Define severity levels
declare -A severity_levels
severity_levels=(
  [DEBUG]=10
  [INFO]=20
  [WARNING]=30
  [ERROR]=40
)

# Print message with time only if level is greater than INFO, to stderr
MINLOGLEVEL="INFO"
print_msg() {
  local message="$1"
  local level=${2:-INFO}

  if [[ ${severity_levels[$level]} -ge ${severity_levels[$MINLOGLEVEL]} ]]; then
    echo >&2 "[$level] $(print_timestamp): $1" # showing messages
  else
    echo "[$level] $(print_timestamp): $message" >&2 # NOT showing messages
  fi
}

# read input (non-silent)
read_input() {
  echo -n "$1"
  read $2
}

# read input silently
read_input_silent() {
  echo -n "$1"
  read -s $2
  echo
}

ask_reset() {
  local varName=${1:-"it"}
  # do you want to reset?
  while true; do
    read_input "Do you want to reset ${varName}? [y/n]: " reset
    case $reset in
    [Yy]*)
      return 1
      break
      ;;
    [Nn]*)
      return 0
      break
      ;;
    *)
      echo "Please answer yes or no."
      ;;
    esac
  done
}

# a function to get file name without the extension
getStemName() {
  local file=$1
  baseName=$(basename $file)
  echo ${baseName%.*}
}

##############################################################################
# CONFIG                                                                     #
##############################################################################

# Set configuration variables
logFile="script.log"
verbose=true
BASE=$(dirname $(realpath $0))

##############################################################################
# INPUT                                                                      #
##############################################################################
# Parse command line options
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --run_name | -n)
    name="$2" # wandb run name
    shift 2
    ;; # past argument and value
  --ckpt | -m)
    ckpt="$2" # checkpoint
    shift 2
    ;; # past argument and value
  --config | -c)
    config="$2" # config file
    shift 2
    ;; # past argument and value
  --outdir | -o)
    outDir="$2" # output directory
    shift 2
    ;; # past argument and value
  --level)
    MINLOGLEVEL="$2"
    shift 2
    ;; # past argument and value
  --help | -h)
    usage
    shift # past argument
    exit 1
    ;;
  *)
    echo "Illegal option: $key"
    usage
    exit 1
    ;;
  esac
done
# if outDir then create directory
[[ -n $outDir ]] && mkdir -p $outDir

# assert required arguments
[[ -z $ckpt ]] && echo "ckpt is not set" && usage && exit 1
[[ -z $config ]] && echo "config is not set" && usage && exit 1

##############################################################################
# MAIN                                                                       #
##############################################################################

# ----------------------------------------
# examine env
# ----------------------------------------
errMessages=()
# assert evaluate_on_walle.py exists
[[ ! -f "$BASE/evaluate_on_walle.py" ]] && errMessages+=("evaluate.py exists ... no")
# assert calculate_metrics.py exists
[[ ! -f "$BASE/calculate-mean-metrics.py" ]] && errMessages+=("calculate_metrics.py exists ... no")
# assert the provided model checkpoint exists
[[ ! -f "$ckpt" ]] && errMessages+=("checkpoint exists ... no")
# assert the provided config file exists
[[ ! -f "$config" ]] && errMessages+=("config exists ... no")
# print error messages
[[ ${#errMessages[@]} -gt 0 ]] && for msg in $errMessages; do print_msg $msg "ERROR"; done && exit 1

# ----------------------------------------
# Evaluate the model
# ----------------------------------------
# Evaluate the model
python evaluate_on_walle.py \
  -o $outDir/$name \
  -c $config \
  -m $ckpt >$outDir/${name}-evaluate.log

# Calculate avg metrics
python calculate-mean-metrics.py \
  ${outDir}/${name} -json ${outDir}/${name}-summary.json
