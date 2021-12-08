
# Extract Vivado HLS include path
VHLS_PATH := $(dir $(shell which vitis_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

CFLAGS = -g -I${VHLS_INC}

all: csim

csim: tb.cc
	@echo "Compiling & simulating cnn on amdpool ..."
	vitis_hls -f csim.tcl

hls:
	@echo "Run Vivado csim and HLS"
	vitis_hls -f syn.tcl

vivado:
	cp ./bitstream_prepare/design_1_wrapper.v ./bnn.prj/solution1/impl/ip/
	cp ./bitstream_prepare/FracNet-CIFAR10.tcl ./bnn.prj/solution1/impl/ip/
	cd ./bnn.prj/solution1/impl/ip/; vivado -mode tcl -source FracNet-CIFAR10.tcl

clean:
	rm -rf bnn *.prj *.log

