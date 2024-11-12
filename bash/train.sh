echo 'Start training...'
python slidelevel_zeroshot_multiprompt.py \
    --task RCC_subtyping \
    --embeddings_dir  /root/MM/data\
    --dataset_split ./data_csvs/tcga_rcc_zeroshot_example.csv \
    --topj 1 5 50 \
    --prompt_file ./prompts/rcc_prompts.json \
    --model_checkpoint ./logs/ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt 