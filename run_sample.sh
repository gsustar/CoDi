python sample.py \
    --expdir $1 \
    --checkpoint $2 \
    --use_fp16 \
    --allow_resizing \
    --allow_ttn \
    --force_resize 1024 \
    --ids 6653 \
    --timestep_respacing 25
