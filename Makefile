.PHONY: clean

CUDA_INC_PATH = /opt/ud/cuda-8.0/include/
CUDA_LIB_PATH = /opt/ud/cuda-8.0/lib64/

OUTPUT_DIR = bin/
SOURCE_DIR = src/

NVCC_FLAGS = -O3 -m64 -Xcompiler=-fPIC -Xptxas=-v
CC_FLAGS = -O3 -m64 -fPIC

SMS ?= 30 35 37 50 52 60
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all: $(OUTPUT_DIR)/gausstransform.o

$(OUTPUT_DIR)%.o: $(SOURCE_DIR)%.cu $(OUTPUT_DIR)
	nvcc $(NVCC_FLAGS) $(GENCODE_FLAGS) -c $< -o $@

$(OUTPUT_DIR):
	mkdir $@

clean:
	rm -rf $(OUTPUT_DIR)



