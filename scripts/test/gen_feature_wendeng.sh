python gen_feature.py \
    --gpu_id 1  \
    --img_dir  /media/megatron-home/dwliang/data/wendeng_res/ \
    --img_list /home/xqt/demo/retrieval_test/wendeng.txt \
    --feature_path /home/xqt/features/v0_wendeng_fc7.fea \
    --net_def /home/xqt/exp/deploy_multi_v0.prototxt \
    --weights /home/xqt/exp_res/v0_iter_34589.caffemodel  \
    --feat fc7
