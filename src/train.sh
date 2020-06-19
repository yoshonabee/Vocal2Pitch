python3 train.py \
    train.json val.json \
    --lr 0.0001 \
    --epochs 500 \
    --dropout 0 \
    --criterion bceloss \
    --batch_size 128 \
    --name one_cnn_end_to_end
