#!/bin/zsh

set -e

# Aim: Train WALLE on AsEPv1.1.0
# Input:
# Output:
# Usage:
# Example:
# Dependencies:

##############################################################################
# FUNCTION                                                                   #
##############################################################################
function usage() {
  #echo "Usage: $(basename $0) --abdbid <abdbid> --ab_chain <ab_chain> --ag_chain <ag_chain> --abdb <abdb> --s3bucket <s3bucket> --outdir <outdir> --clean <clean>"
  #echo "  --abdbid   : AbDb id"
  #echo "  --ab_chain : Ab chain ids, if multiple, wrap in quotes e.g. 'H L' (default: H L)"
  #echo "  --ag_chain : Ag chain id"
  #echo "  --abdb     : AbDb directory (default: /home/ubuntu/AbDb/asepv1-1724)"
  #echo "  --outdir   : Output directory (default: $PWD)"
  #echo "  --clean    : Clean up the output directory (default: false)"
  #echo "  --s3bucket : (optional) S3 bucket to upload the output"
  #echo "  --esm2_ckpt_host_path : (optional) ESM2 checkpoint host path (default: /home/ubuntu/trained_models/ESM2)"
  #echo "  --help     : Print this help message"
  # add example
  echo "Example:"
  #echo "$(basename $0) --abdbid 7djz_0P --ab_chain 'H L' --ag_chain C --abdb /mnt/Data/AbDb/abdb_newdata_20220926 --outdir ./test  --esm2_ckpt_host_path /mnt/Data/trained_models/ESM2"
  exit 1
}

# Function to print timestamp
print_timestamp() {
  date +"%Y%m%d-%H%M%S"  # e.g. 20240318-085729
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
    >&2 echo "[$level] $(print_timestamp): $1"        # showing messages
  else
    echo "[$level] $(print_timestamp): $message" >&2  # NOT showing messages
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
      [Yy]* )
        return 1
        break
        ;;
      [Nn]* )
        return 0
        break
        ;;
      * )
        echo "Please answer yes or no."
        ;;
    esac
  done
}

# a function to get file name without the extension
function getStemName() {
  local file=$1
  baseName=$(basename $file)
  echo ${baseName%.*}
}

# turn array into json string
turn_array_into_json_string() {
  local array=($@)
  # if array is empty, return empty string
  if [ ${#array[@]} -eq 0 ]; then
    echo "[]"
    return
  fi
  # loop through array and append to json string
  local jsonStr='['
  for t in "${array[@]}"; do
    jsonStr+='"'$t'",'
  done
  jsonStr=${jsonStr%,} # remove trailing comma
  jsonStr+=']'
  echo $jsonStr
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
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --mode|-m)
      runMode="$2"
      shift 2;; # past argument and value
    --notes|-n)
      runNotes="$2"
      shift 2;; # past argument and value
    --run_tags|-t)
      runTags=() # initialize as empty array
      shift # move past the argument key
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        runTags+=("$1") # append tag to array
        shift # move to next argument
      done;;
    --level)
      MINLOGLEVEL="$2"
      shift 2;; # past argument and value
    --help|-h)
      usage
      shift # past argument
      exit 1;;
    *)
      echo "Illegal option: $key"
      usage
      exit 1;;
  esac
done

##############################################################################
# MAIN                                                                       #
##############################################################################
projectName=${WANDB_PROJECT:-"retrain-walle"}
runMode=${runMode:-"dev"}
runNotes=${runNotes:-"Retrain WALLE on AsEPv1.1.0"}
runTagStr=$(turn_array_into_json_string "${runTags[@]}")

python train.py \
  mode=$runMode \
  "wandb_init.project=\"$projectName\"" \
  "wandb_init.notes=$runNotes" \
  "wandb_init.tags=$runTagStr" \
  hparams.max_epochs=200 \
  hparams.pos_weight=100 \
  hparams.train_batch_size=128 \
  hparams.val_batch_size=32 \
  hparams.test_batch_size=32
