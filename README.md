# cuda_fft_is_slow
If an FFT is the only thing you are gonna do with your data, you may as well just do it on your CPU...

This is the lesson I learned while coding up this GPU based FFT algorithm. The actual processing is, as you would expect, incredibly fast. 
But it accounts for only about 1% of 1% of the entire runtime of the whole GPU FFT procedure.
The remainder of the time is spent copying the data of interest into VRAM from DRAM. 
Moral? Do more than just an FFT with your data. 

### Benchmarking

#### 2^24 point FFT
| Alg | Time |
| --- | ---  |
| GPU | 547 ms |
| GPU (excluding memory ops) | 67 us |
| CPU (numpy) | 971 ms |
| CPU (mine) | 17 s |
