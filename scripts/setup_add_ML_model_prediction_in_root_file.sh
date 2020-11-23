#---Source this script to setup the complete environment for this LCG view
#   Generated: Fri Sep 18 10:28:42 2020

#---Get the location this script (thisdir)
# SOURCE=${BASH_ARGV[0]}
# if [ "x$SOURCE" = "x" ]; then
#     SOURCE=${(%):-%N} # for zsh
# fi
# thisdir=$(cd "$(dirname "${SOURCE}")"; pwd)
# # Note: readlink -f is not working on OSX
# thisdir=$(python -c "import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))" ${thisdir})
thisdir=/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt/

#  First the compiler
if [ "$COMPILER" != "native" ] && [ -e /cvmfs/sft.cern.ch/lcg/releases/gcc/9.2.0/x86_64-centos7/setup.sh ]; then
    source /cvmfs/sft.cern.ch/lcg/releases/gcc/9.2.0/x86_64-centos7/setup.sh
fi

#  LCG version
LCG_VERSION=98python3; export LCG_VERSION

#  then the rest...
if [ -z "${PATH}" ]; then
    PATH=${thisdir}/bin; export PATH
else
    PATH=${thisdir}/bin:$PATH; export PATH
fi
if [ -d ${thisdir}/scripts ]; then
    PATH=${thisdir}/scripts:$PATH; export PATH
fi

if [ -z "${LD_LIBRARY_PATH}" ]; then
    LD_LIBRARY_PATH=${thisdir}/lib; export LD_LIBRARY_PATH
else
    LD_LIBRARY_PATH=${thisdir}/lib:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
fi
if [ -d ${thisdir}/lib64 ]; then
    LD_LIBRARY_PATH=${thisdir}/lib64:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
fi

PYTHON_VERSION=`expr $(readlink ${thisdir}/bin/python) : '.*/Python/\([0-9].[0-9]\).*'`
PY_PATHS=${thisdir}/lib/python$PYTHON_VERSION/site-packages

if [ -z "${PYTHONPATH}" ]; then
    PYTHONPATH=${thisdir}/lib:$PY_PATHS; export PYTHONPATH
else
    PYTHONPATH=${thisdir}/lib:$PY_PATHS:$PYTHONPATH; export PYTHONPATH
fi
if [ -d ${thisdir}/python ]; then
    PYTHONPATH=${thisdir}/python:$PYTHONPATH; export PYTHONPATH
fi
# 
# if [ -z "${MANPATH}" ]; then
#     MANPATH=${thisdir}/man:${thisdir}/share/man; export MANPATH
# else
#     MANPATH=${thisdir}/man:${thisdir}/share/man:$MANPATH; export MANPATH
# fi
# if [ -z "${CMAKE_PREFIX_PATH}" ]; then
#     CMAKE_PREFIX_PATH=${thisdir}; export CMAKE_PREFIX_PATH
# else
#     CMAKE_PREFIX_PATH=${thisdir}:$CMAKE_PREFIX_PATH; export CMAKE_PREFIX_PATH
# fi
# if [ -z "${CPLUS_INCLUDE_PATH}" ]; then
#     CPLUS_INCLUDE_PATH=${thisdir}/include; export CPLUS_INCLUDE_PATH
# else
#     CPLUS_INCLUDE_PATH=${thisdir}/include:$CPLUS_INCLUDE_PATH; export CPLUS_INCLUDE_PATH
# fi
# if [ -z "${C_INCLUDE_PATH}" ]; then
#     C_INCLUDE_PATH=${thisdir}/include; export C_INCLUDE_PATH
# else
#     C_INCLUDE_PATH=${thisdir}/include:$C_INCLUDE_PATH; export C_INCLUDE_PATH
# fi
#if [ -z "${LIBRARY_PATH}" ]; then
#    LIBRARY_PATH=${thisdir}/lib; export LIBRARY_PATH
#else
#    LIBRARY_PATH=${thisdir}/lib:$LIBRARY_PATH; export LIBRARY_PATH
#fi
#if [ -d ${thisdir}/lib64 ]; then
#    export LIBRARY_PATH=${thisdir}/lib64:$LIBRARY_PATH
#fi

# #---check for compiler variables
# if [ -z "${CXX}" ]; then
#     export FC=`command -v gfortran`
#     export CC=`command -v gcc`
#     export CXX=`command -v g++`
# fi
# 
# #---Figure out the CMAKE_CXX_STANDARD (using Vc as a victim)
# if [ -f $thisdir/include/Vc/Vc ]; then
#     vc_home=$(dirname $(dirname $(dirname $(readlink $thisdir/include/Vc/Vc))))
#     std=$(cat $vc_home/logs/Vc*configure.cmake | egrep -o "CMAKE_CXX_STANDARD=[0-9]+" | egrep -o "[0-9]+")
#     export CMAKE_CXX_STANDARD=$std
# fi

# #---then ROOT
# if [ -x $thisdir/bin/root ]; then
#     PYTHON_INCLUDE_PATH=$(dirname $(dirname $(readlink $thisdir/bin/python)))/include/$(\ls $(dirname $(dirname $(readlink $thisdir/bin/python)))/include)
#     ROOTSYS=$(dirname $(dirname $(readlink $thisdir/bin/root))); export ROOTSYS
#     if [ -z "${ROOT_INCLUDE_PATH}" ]; then
#         ROOT_INCLUDE_PATH=${thisdir}/include:$PYTHON_INCLUDE_PATH; export ROOT_INCLUDE_PATH
#     else
#         ROOT_INCLUDE_PATH=${thisdir}/include:$PYTHON_INCLUDE_PATH:$ROOT_INCLUDE_PATH; export ROOT_INCLUDE_PATH
#     fi
#     if [ -d $thisdir/targets/x86_64-linux/include ]; then
#         ROOT_INCLUDE_PATH=${thisdir}/targets/x86_64-linux/include:$ROOT_INCLUDE_PATH; export ROOT_INCLUDE_PATH
#     fi
#     if [ -z "${JUPYTER_PATH}" ]; then
#         JUPYTER_PATH=${thisdir}/etc/notebook; export JUPYTER_PATH
#     else
#         JUPYTER_PATH=${thisdir}/etc/notebook:$JUPYTER_PATH; export JUPYTER_PATH
#     fi
#     export CPPYY_BACKEND_LIBRARY=$ROOTSYS/lib/libcppyy_backend${PYTHON_VERSION/./_}
#     export CLING_STANDARD_PCH=none
# fi

# #---then PYTHON
# if [ -x $thisdir/bin/python ]; then
#     PYTHONHOME=$(dirname $(dirname $(readlink $thisdir/bin/python))); export PYTHONHOME
# elif [ -x $thisdir/bin/python3 ]; then
#     PYTHONHOME=$(dirname $(dirname $(readlink $thisdir/bin/python3))); export PYTHONHOME
# fi
# 
# #---then tensorflow
# if [ -d $PY_PATHS/tensorflow_core ]; then   # version > 2.0
#     export LD_LIBRARY_PATH=$PY_PATHS/tensorflow_core:$LD_LIBRARY_PATH
# 
# elif [ -d $PY_PATHS/tensorflow ]; then
#     export LD_LIBRARY_PATH=$PY_PATHS/tensorflow:$PY_PATHS/tensorflow/contrib/tensor_forest:$PY_PATHS/tensorflow/python/framework:$LD_LIBRARY_PATH
# fi

# #---then PKG_CONFIG_PATH
# if [ -z "$PKG_CONFIG_PATH" ]; then
#     export PKG_CONFIG_PATH="$thisdir/lib/pkgconfig"
# else
#     export PKG_CONFIG_PATH="$thisdir/lib/pkgconfig:$PKG_CONFIG_PATH"
# fi
# if [ -d ${thisdir}/lib64/pkgconfig ]; then
#     PKG_CONFIG_PATH=${thisdir}/lib64/pkgconfig:$PKG_CONFIG_PATH; export PKG_CONFIG_PATH
# fi
# 
