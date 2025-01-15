export CUDA_VISIBLE_DEVICES=2

for model in gwnet # ttg_iso dcrnn gwnet agcrn
do
    for data in la bay air # cere
    do
        for emb in uniform none
        do
            python -m experiments.run_ckpt config=benchmarks model=$model embedding=$emb dataset=$data &
        done
        wait
    done
done
