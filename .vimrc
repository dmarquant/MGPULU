set expandtab
set softtabstop=4
set shiftwidth=4

set smartindent

com! Test !CUDA_VISIBLE_DEVICES=0 srun -w octane008 ./lu


