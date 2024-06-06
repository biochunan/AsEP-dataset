.PHONY: setup-gpu-env setup-cpu-env

install-tools:
	sudo apt update && sudo apt install htop nvtop tmux parallel bc -y

setup-gpu-env:
	conda env update -f conda-env/conda-env-gpu.yaml

setup-cpu-env:
	conda env update -f conda-env/conda-env-cpu.yaml

install-dependencies:
	pip install -Ue .
	conda install -y bioconda::mmseqs2
