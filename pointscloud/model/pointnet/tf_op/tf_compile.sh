#!/bin/bash
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo ${TF_CFLAGS[@]}
nvcc -std=c++11 -c -o tf_op_cu.o tf_op.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o tf_op_cu.so tf_op.cpp tf_op_cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -I /usr/local/cuda-10.2/include -L /usr/local/cuda-10.2/lib64

rm -rf ../../lib/*
cp tf_op_cu.so ../../lib/
