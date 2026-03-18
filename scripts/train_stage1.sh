python -m main.cli.train_stage1 \
    --model_name_or_path /mnt/dhwfile/raise/user/panjiabao/huxiaobin/shy/ckpt/model/Qwen2.5-VL-3B-Instruct \
    --train_jsonl /mnt/dhwfile/raise/user/panjiabao/huxiaobin/shy/data/Video-R1-data/Video-R1-COT-165k.json \
    --output_dir /mnt/dhwfile/raise/user/panjiabao/huxiaobin/shy/ckpt/VideoUnderStand/stage1

# CUDA_VISIBLE_DEVICES=4,5 