python brute_force.py \
    --gpu_id 0 \
    --query_dir  /media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/ \
    --query_list /home/xqt/demo/retrieval_test/query_3.txt \
    --siyang_label /media/megatron-home/dwliang/search_eval/query/cropped_label/ \
    --nn_number 1000 \
    --net_def /home/xqt/exp/deploy_multi_v0.prototxt \
    --weights /home/xqt/exp_res/v0_iter_34589.caffemodel \
    --feat fc7 \
    --begin_loc 1 \
    --end_loc 2049 \
    --siyang_feat /home/xqt/features/v0_siyang_fc7.fea \
    --wendeng_feat /home/xqt/features/v0_wendeng_fc7.fea
