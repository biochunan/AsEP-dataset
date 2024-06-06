#!/bin/zsh

set -e

##############################################################################
# FUNCTION                                                                   #
##############################################################################
usage() {
  echo "Usage: $(basename $0) --abdbid|-i <abdbid> --job_name|-o <job_name> --outdir|-o <outdir> --abdb|-d <abdb> [--level <level>] [--help|-h]"
  echo "Options:"
  echo "  --abdbid  |-i   <abdbid>   : AbDb id e.g. 1a14_0P"
  echo "  --job_name|-n   <job_name> : Job name"
  echo "  --outdir  |-o   <outdir>   : Output directory"
  echo "  --abdb    |-d   <abdb>     : AbDb directory"
  echo "  --level   |-l   <level>    : Log level (DEBUG, INFO, WARNING, ERROR)"
  echo "  --help    |-h  Help"
  # add example
  echo "Example:"
  echo "  $(basename $0) --abdbid 1a14_0P --job_name 1a14_N --outdir /path/to/output --abdb /path/to/abdb"
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
outDir=$BASE/masif_output
# Parse command line options
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --abdbid|-i)
      abdbid="$2"
      shift 2;; # past argument and value
    --job_name|-n)
      jobName="$2"
      shift 2;; # past argument and value
    --outdir|-o)
      outDir="$2"
      shift 2;; # past argument and value
    --abdb|-d)
      ABDB="$2"
      shift 2;; # past argument and value
    --level|-l)
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
# if outDir then create directory
[[ -n $outDir ]] && mkdir -p $outDir

# assert required arguments
[[ -z $abdbid ]] && print_msg "AbDb id is required" "ERROR" && usage
print_msg "abdbid is provided ........... yes"
[[ -z $jobName ]] && print_msg "Job name is required" "ERROR" && usage
print_msg "job name is provided ......... yes"
[[ -z $ABDB ]] && print_msg "AbDb directory is required" "ERROR" && usage
print_msg "AbDb directory is provided ... yes"
[[ ! -d $ABDB ]] && print_msg "AbDb directory does not exist" "ERROR" && usage
print_msg "AbDb directory exists ........ yes"

##############################################################################
# MAIN                                                                       #
##############################################################################
# make dirs
mkdir -p $outDir/data_preparation
mkdir -p $outDir/pred_output
mkdir -p $outDir/logs/$jobName

# `/dataset/abdb`: for the input structure files
# `/masif/data/masif_site/data_preparation`: location of preprocessed files for inference input
# `/masif/data/masif_site`: location of preprocessed files for training input

docker run --rm \
  -v $ABDB:/dataset/abdb \
  -v $outDir/data_preparation:/masif/data/masif_site/data_preparation  \
  -v $outDir/pred_output:/masif/data/masif_site/output  \
  -w /masif/data/masif_site \
  pablogainza/masif \
  /masif/data/masif_site/data_prepare_one.sh  \
  --file /dataset/abdb/pdb${abdbid}.mar ${jobName} \
  > $outDir/logs/$jobName/data_prepare_one.log 2>&1

# run predict_site
docker run --rm \
  -v $ABDB:/dataset/abdb \
  -v $outDir/data_preparation:/masif/data/masif_site/data_preparation  \
  -v $outDir/pred_output:/masif/data/masif_site/output  \
  -w /masif/data/masif_site \
  pablogainza/masif \
  /masif/data/masif_site/predict_site.sh ${jobName} \
  > $outDir/logs/$jobName/predict_site.log 2>&1

# run color_site.sh
docker run --rm \
  -v $ABDB:/dataset/abdb \
  -v $outDir/data_preparation:/masif/data/masif_site/data_preparation  \
  -v $outDir/pred_output:/masif/data/masif_site/output  \
  -w /masif/data/masif_site \
  pablogainza/masif \
  /masif/data/masif_site/color_site.sh ${jobName} \
  > $outDir/logs/$jobName/color_site.log 2>&1
