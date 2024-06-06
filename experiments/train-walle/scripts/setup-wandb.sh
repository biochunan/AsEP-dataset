#!/bin/zsh

# Aim: Set environment variables for WALLE on AsEPv1.0.0

# shellVersion=$(basename $SHELL)

##############################################################################
# FUNCTION                                                                   #
##############################################################################
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

  if [[ ${severity_levels[$level]} -ge ${severity_levels["$MINLOGLEVEL"]} ]]; then
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

set_api_key() {
  read_input_silent "Enter your Weights & Biases API key (obtain from: https://wandb.ai/authorize):" WANDB_API_KEY
  export WANDB_API_KEY=$WANDB_API_KEY
  echo "export WANDB_API_KEY='$WANDB_API_KEY'" >> $tempFile
}

set_entity() {
  read_input "Enter your Weights & Biases entity name [default to $(whoami)]: " WANDB_ENTITY
  export WANDB_ENTITY=$WANDB_ENTITY
  echo "export WANDB_ENTITY='$WANDB_ENTITY'" >> $tempFile
}

set_project() {
  now=$(date +"%Y%m%d-%H%M%S")
  read_input "Set Weights & Biases project name [default to '$now']: " WANDB_PROJECT
  WANDB_PROJECT=${WANDB_PROJECT:-$now}
  export WANDB_PROJECT=$WANDB_PROJECT
  echo "export WANDB_PROJECT='$WANDB_PROJECT'" >> $tempFile
}

set_mode(){
  # WNADB mode default to online, choices: online, offline, disabled
  valid_modes=("online" "offline" "disabled")
  while true; do
    read_input "Set Weights & Biases mode [online(default)/offline/disabled]: " WANDB_MODE
    WANDB_MODE=${WANDB_MODE:-online}
    if [[ " ${valid_modes[@]} " =~ " ${WANDB_MODE} " ]]; then
      export WANDB_MODE=${WANDB_MODE:-online}
      echo "export WANDB_MODE='$WANDB_MODE'" >> $tempFile
      break 2
    else
      echo "Invalid mode. Please enter a valid mode."
    fi
  done

}

ask_to_write_to_shell_config() {
  local localConfigFile=$1
  while true; do
    rcFile=$HOME/.zshrc
    read_input "Do you want to write the setup to $rcFile? [y/n]: " write
    case $write in
      [Yy]* )
        print_msg "Writing the setup to $rcFile ..." "INFO"
        echo "source $localConfigFile" >> $HOME/.zshrc
        print_msg "Done." "INFO"
        break
        ;;
      [Nn]* )
        return 0
        ;;
      * )
        echo "Please answer y or n."
        ;;
    esac
  done
}

##############################################################################
# MAIN                                                                       #
##############################################################################
BASE=$(dirname $(realpath $0))

# final config file
configFile=$(dirname $BASE)/envs/env-setup.sh
mkdir -p $(dirname $configFile)

# check if the config file exists
if [[ -f $configFile ]]; then
  print_msg "Environment setup file already exists: $configFile" "INFO"
  ask_reset "the environment setup file"
  if [[ $? == 0 ]]; then
    source $configFile
    ask_to_write_to_shell_config $configFile
    print_msg "Exit" "INFO"
    return 0
  else
    print_msg "Overwriting the environment setup file..." "INFO"
  fi
fi

# make a temp config file
tempFile=$(mktemp)
print_msg "Temporary file created: $tempFile" "INFO"
# set a trap
trap "print_msg 'Clean up...' 'INFO'; rm -f $tempFile; print_msg 'Done.' 'INFO' " EXIT INT TERM

if [[ -z "$(echo ${WANDB_API_KEY})" ]]; then
  set_api_key
else
  print_msg "Weights & Biases API key is already set."
  ask_reset "Weights & Biases API key"
  if [[ $? == 1 ]]; then
    set_api_key
  fi
fi

if [[ -z "$(echo ${WANDB_ENTITY})" ]]; then
  set_entity
else
  print_msg "Weights & Biases entity name is already set."
  ask_reset "Weights & Biases entity name"
  if [[ $? == 1 ]]; then
    set_entity
  fi
fi

# optional setting: PROJECT, MODE
if [[ -z "$(echo ${WANDB_PROJECT})" ]]; then
  set_project
else
  print_msg "Weights & Biases project name is already set."
  ask_reset "Weights & Biases project name"
  if [[ $? == 1 ]]; then
    set_project
  fi
fi

# WNADB mode
if [[ -z "$(echo ${WANDB_MODE})" ]]; then
  set_mode
else
  print_msg "Weights & Biases mode is already set."
  ask_reset "Weights & Biases mode"
  if [[ $? == 1 ]]; then
    set_mode
  fi
fi

# copy the temp file to the final config file
cat $tempFile > $configFile
ask_to_write_to_shell_config $configFile