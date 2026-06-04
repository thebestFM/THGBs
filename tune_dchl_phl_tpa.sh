#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-logs_tuning/dchl_phl_tpa}"
mkdir -p "$LOG_ROOT"

run_exp() {
  name="$1"
  shift
  log_file="$LOG_ROOT/${name}.log"
  echo "[DCHL tune] start ${name} -> ${log_file}"
  "$PYTHON_BIN" "$@" --save "$name" > "$log_file" 2>&1 &
  pid=$!
  wait "$pid"
  echo "[DCHL tune] done  ${name}"
}

COMMON="train_dchl_fair.py --gpu 0 --seed 42 --ns_q 1000 --ns_seed 42 --train_predict_ratio 0.0 --progress_every 200 --tolerance 1e-8"
PHL="$COMMON --dataset Yelp-PHL --batch_size 384 --eval_batch_size 768 --evaluate_every 5 --patience 35"
TPA="$COMMON --dataset Yelp-TPA --batch_size 512 --eval_batch_size 1024 --evaluate_every 5 --patience 40"

run_exp phl_dchl_d64_l222_cl003_fw004_tw15_geo15 $PHL --n_epochs 70 --emb_dim 64 --num_mv_layers 2 --num_geo_layers 2 --num_di_layers 2 --dropout 0.35 --lambda_cl 0.03 --temperature 0.1 --lr 0.001 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --distance_threshold 1.5 --transition_window 15 --friend_weight 0.04 --query_score_weight 1.0 --time_score_weight 0.10 --num_time_bins 128
run_exp phl_dchl_d96_l222_cl003_fw006_tw20_geo25 $PHL --n_epochs 70 --emb_dim 96 --num_mv_layers 2 --num_geo_layers 2 --num_di_layers 2 --dropout 0.40 --lambda_cl 0.03 --temperature 0.1 --lr 0.0007 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --distance_threshold 2.5 --transition_window 20 --friend_weight 0.06 --query_score_weight 1.0 --time_score_weight 0.10 --num_time_bins 128
run_exp phl_dchl_d96_l322_cl005_fw008_tw20_geo25 $PHL --n_epochs 70 --emb_dim 96 --num_mv_layers 3 --num_geo_layers 2 --num_di_layers 2 --dropout 0.40 --lambda_cl 0.05 --temperature 0.1 --lr 0.0007 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --distance_threshold 2.5 --transition_window 20 --friend_weight 0.08 --query_score_weight 1.0 --time_score_weight 0.15 --num_time_bins 256
run_exp phl_dchl_d128_l222_cl003_fw004_tw30_geo25_bs256 $COMMON --dataset Yelp-PHL --batch_size 256 --eval_batch_size 512 --evaluate_every 5 --patience 35 --n_epochs 60 --emb_dim 128 --num_mv_layers 2 --num_geo_layers 2 --num_di_layers 2 --dropout 0.45 --lambda_cl 0.03 --temperature 0.1 --lr 0.0005 --decay 0.001 --grad_norm 1.0 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.5 --distance_threshold 2.5 --transition_window 30 --friend_weight 0.04 --query_score_weight 1.2 --time_score_weight 0.15 --num_time_bins 256
run_exp phl_dchl_d96_l333_cl010_fw006_tw30_geo50 $PHL --n_epochs 60 --emb_dim 96 --num_mv_layers 3 --num_geo_layers 3 --num_di_layers 3 --dropout 0.45 --lambda_cl 0.10 --temperature 0.07 --lr 0.0005 --decay 0.001 --grad_norm 1.0 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.5 --distance_threshold 5.0 --transition_window 30 --friend_weight 0.06 --query_score_weight 1.0 --time_score_weight 0.15 --num_time_bins 256
run_exp phl_dchl_d64_l322_cl020_fw004_tw20_geo10 $PHL --n_epochs 70 --emb_dim 64 --num_mv_layers 3 --num_geo_layers 2 --num_di_layers 2 --dropout 0.40 --lambda_cl 0.20 --temperature 0.05 --lr 0.001 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --distance_threshold 1.0 --transition_window 20 --friend_weight 0.04 --query_score_weight 1.0 --time_score_weight 0.10 --num_time_bins 128
run_exp phl_dchl_d96_l222_nofriend_tw20_geo25 $PHL --n_epochs 60 --emb_dim 96 --num_mv_layers 2 --num_geo_layers 2 --num_di_layers 2 --dropout 0.40 --lambda_cl 0.05 --temperature 0.1 --lr 0.0007 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.5 --distance_threshold 2.5 --transition_window 20 --no_friend_edges --query_score_weight 1.0 --time_score_weight 0.10 --num_time_bins 128
run_exp phl_dchl_d128_l322_cl005_fw008_tw20_geo25_bs256 $COMMON --dataset Yelp-PHL --batch_size 256 --eval_batch_size 512 --evaluate_every 5 --patience 35 --n_epochs 60 --emb_dim 128 --num_mv_layers 3 --num_geo_layers 2 --num_di_layers 2 --dropout 0.45 --lambda_cl 0.05 --temperature 0.1 --lr 0.0005 --decay 0.001 --grad_norm 1.0 --lr_scheduler cosine --distance_threshold 2.5 --transition_window 20 --friend_weight 0.08 --query_score_weight 1.2 --time_score_weight 0.15 --num_time_bins 256

run_exp tpa_dchl_d96_l222_cl003_fw008_tw15_geo15 $TPA --n_epochs 90 --emb_dim 96 --num_mv_layers 2 --num_geo_layers 2 --num_di_layers 2 --dropout 0.30 --lambda_cl 0.03 --temperature 0.1 --lr 0.001 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --distance_threshold 1.5 --transition_window 15 --friend_weight 0.08 --query_score_weight 1.0 --time_score_weight 0.10 --num_time_bins 128
run_exp tpa_dchl_d96_l322_cl005_fw010_tw20_geo25 $TPA --n_epochs 90 --emb_dim 96 --num_mv_layers 3 --num_geo_layers 2 --num_di_layers 2 --dropout 0.35 --lambda_cl 0.05 --temperature 0.1 --lr 0.0008 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --distance_threshold 2.5 --transition_window 20 --friend_weight 0.10 --query_score_weight 1.0 --time_score_weight 0.15 --num_time_bins 256
run_exp tpa_dchl_d128_l322_cl005_fw015_tw20_geo25 $TPA --n_epochs 90 --emb_dim 128 --num_mv_layers 3 --num_geo_layers 2 --num_di_layers 2 --dropout 0.35 --lambda_cl 0.05 --temperature 0.1 --lr 0.0005 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --distance_threshold 2.5 --transition_window 20 --friend_weight 0.15 --query_score_weight 1.2 --time_score_weight 0.15 --num_time_bins 256
run_exp tpa_dchl_d128_l333_cl010_fw015_tw30_geo50 $TPA --n_epochs 80 --emb_dim 128 --num_mv_layers 3 --num_geo_layers 3 --num_di_layers 3 --dropout 0.40 --lambda_cl 0.10 --temperature 0.07 --lr 0.0005 --decay 0.001 --grad_norm 1.0 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --distance_threshold 5.0 --transition_window 30 --friend_weight 0.15 --query_score_weight 1.0 --time_score_weight 0.15 --num_time_bins 256
run_exp tpa_dchl_d160_l322_cl003_fw010_tw30_geo25_bs384 $COMMON --dataset Yelp-TPA --batch_size 384 --eval_batch_size 768 --evaluate_every 5 --patience 40 --n_epochs 80 --emb_dim 160 --num_mv_layers 3 --num_geo_layers 2 --num_di_layers 2 --dropout 0.40 --lambda_cl 0.03 --temperature 0.1 --lr 0.0004 --decay 0.001 --grad_norm 1.0 --lr_scheduler cosine --distance_threshold 2.5 --transition_window 30 --friend_weight 0.10 --query_score_weight 1.2 --time_score_weight 0.20 --num_time_bins 256
run_exp tpa_dchl_d96_l322_cl020_fw008_tw20_geo10 $TPA --n_epochs 90 --emb_dim 96 --num_mv_layers 3 --num_geo_layers 2 --num_di_layers 2 --dropout 0.35 --lambda_cl 0.20 --temperature 0.05 --lr 0.0008 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --distance_threshold 1.0 --transition_window 20 --friend_weight 0.08 --query_score_weight 1.0 --time_score_weight 0.10 --num_time_bins 128
run_exp tpa_dchl_d128_l222_nofriend_tw20_geo25 $TPA --n_epochs 80 --emb_dim 128 --num_mv_layers 2 --num_geo_layers 2 --num_di_layers 2 --dropout 0.35 --lambda_cl 0.05 --temperature 0.1 --lr 0.0005 --decay 0.0005 --grad_norm 1.0 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --distance_threshold 2.5 --transition_window 20 --no_friend_edges --query_score_weight 1.0 --time_score_weight 0.10 --num_time_bins 128
run_exp tpa_dchl_d128_l423_cl005_fw012_tw40_geo25_bs384 $COMMON --dataset Yelp-TPA --batch_size 384 --eval_batch_size 768 --evaluate_every 5 --patience 40 --n_epochs 80 --emb_dim 128 --num_mv_layers 4 --num_geo_layers 2 --num_di_layers 3 --dropout 0.40 --lambda_cl 0.05 --temperature 0.1 --lr 0.0005 --decay 0.001 --grad_norm 1.0 --lr_scheduler cosine --distance_threshold 2.5 --transition_window 40 --friend_weight 0.12 --query_score_weight 1.2 --time_score_weight 0.20 --num_time_bins 256
