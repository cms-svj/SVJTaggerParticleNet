#!/usr/bin/env bash

case `uname` in
  Linux) ECHO="echo -e" ;;
  *) ECHO="echo" ;;
esac

usage(){
	EXIT=$1
	$ECHO "setup.sh [options]"
	$ECHO
	$ECHO "Options:"
	$ECHO "-d              \tuse the developer branch of Coffea (default = 0)"
	$ECHO "-h              \tprint this message and exit"
	$ECHO "-n [NAME]       \toverride the name of the virtual environment"
	exit $EXIT
}

NAME=training
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_99cuda/x86_64-centos7-gcc8-opt
DEV=0

# check arguments
while getopts "dhn:" opt; do
	case "$opt" in
		d) DEV=1
		;;
		h) usage 0
		;;
		n) NAME=$OPTARG
		;;
		:) printf "missing argument for -%s\n" "$OPTARG" >&2
		   usage -1
		;;
		\?) printf "illegal option: -%s\n" "$OPTARG" >&2
		    usage -2
		;;
	esac
done

# Setup the LCG environment
$ECHO "Getting the LCG environment ... "
source $LCG/setup.sh

# Install most of the needed software in a virtual environment
# following https://aarongorka.com/blog/portable-virtualenv/, an alternative is https://github.com/pantsbuild/pex
$ECHO "\nMaking and activating the virtual environment ... "
python -m venv --copies $NAME
source $NAME/bin/activate
$ECHO "\nInstalling 'pip' packages ... "
python -m pip install --no-cache-dir setuptools pip argparse --upgrade 
python -m pip install --no-cache-dir xxhash
python -m pip install --no-cache-dir uproot4
python -m pip install --no-cache-dir magiconfig
if [[ "$DEV" == "1" ]]; then
	$ECHO "\nInstalling the 'development' version of Coffea ... "
	python -m pip install --no-cache-dir flake8 pytest coverage
	git clone https://github.com/CoffeaTeam/coffea
	cd coffea
	python -m pip install --no-cache-dir --editable .[dask,spark,parsl] 'uproot-methods<0.9.0,>=0.7.3' 'pillow>=7.1.0' 'mplhep==0.1.35'
	cd ..
else
	$ECHO "Installing the 'production' version of Coffea ... "
	python -m pip install --no-cache-dir coffea[dask,spark,parsl] 'uproot-methods<0.9.0,>=0.7.3' 'pillow>=7.1.0' 'mplhep==0.1.35'
fi

# Setup the activation script for the virtual environment
$ECHO "\nSetting up the activation script for the virtual environment ... "
sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
find ${NAME}/bin/ -type f -print0 | xargs -0 -P 4 sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/'
sed -i "2a source ${LCG}/setup.sh"'\nexport PYTHONPATH=""' $NAME/bin/activate
sed -i "4a source ${LCG}/setup.csh"'\nsetenv PYTHONPATH ""' $NAME/bin/activate.csh

$ECHO "\nSetting up the ipython/jupyter kernel ... "
storage_dir=$(readlink -f $PWD)
ipython kernel install --prefix=${storage_dir}/.local --name=$NAME
tar -zcf ${NAME}.tar.gz ${NAME}

deactivate
$ECHO "\nFINISHED"
