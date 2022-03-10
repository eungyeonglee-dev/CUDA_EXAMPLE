#NVCC=nvcc
NVCC=/usr/local/cuda/bin/nvcc
SOURCE=mat_mul.cu

exe: $(SOURCE)
	$(NVCC) -o $@ $(SOURCE) ../Timer/Timer.cpp
clean:
	rm -f exe
