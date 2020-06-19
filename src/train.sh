python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.0001 \
    --epochs 150 \
    --dropout 0 \
    --criterion bceloss \
    --batch_size 256 \
    --name big_kernel_size_stride_16_16_2
