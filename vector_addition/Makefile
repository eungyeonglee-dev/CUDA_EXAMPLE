#NVCC=nvcc
NVCC=/usr/local/cuda/bin/nvcc
SOURCE=vec_add.cu

exe: $(SOURCE)
	$(NVCC) -o $@ $(SOURCE) ../Timer/Timer.cpp
clean:
	rm -f exe
