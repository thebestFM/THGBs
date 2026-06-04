#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-logs_tuning/agran_phl_tpa}"
mkdir -p "$LOG_ROOT"

run_exp() {
  name="$1"
  shift
  log_file="$LOG_ROOT/${name}.log"
  echo "[AGRAN tune] start ${name} -> ${log_file}"
  "$PYTHON_BIN" "$@" --save "$name" > "$log_file" 2>&1 &
  pid=$!
  wait "$pid"
  echo "[AGRAN tune] done  ${name}"
}

COMMON="train_agran_fair.py --gpu 0 --seed 42 --ns_q 1000 --ns_seed 42 --train_predict_ratio 0.0 --progress_every 200 --grad_norm 1.0 --time_bucket_seconds 86400"
PHL="$COMMON --dataset Yelp-PHL --batch_size 256 --eval_batch_size 512 --evaluate_every 5 --patience 30"
TPA="$COMMON --dataset Yelp-TPA --batch_size 384 --eval_batch_size 768 --evaluate_every 5 --patience 35"

run_exp phl_agran_h64_l40_b2_kl02_fp002 $PHL --n_epochs 70 --hidden_units 64 --maxlen 40 --num_blocks 2 --num_heads 2 --dropout_rate 0.35 --kl_reg 0.2 --lr 0.001 --l2_emb 0.0001 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.12 --rel_bias_weight 1.0 --friend_prior_weight 0.02 --friend_prior_topk 2
run_exp phl_agran_h64_l50_b3_kl02_fp002 $PHL --n_epochs 70 --hidden_units 64 --maxlen 50 --num_blocks 3 --num_heads 2 --dropout_rate 0.35 --kl_reg 0.2 --lr 0.001 --l2_emb 0.0001 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.02 --friend_prior_topk 2
run_exp phl_agran_h96_l40_b2_kl02_fp002 $PHL --n_epochs 60 --hidden_units 96 --maxlen 40 --num_blocks 2 --num_heads 3 --dropout_rate 0.40 --kl_reg 0.2 --lr 0.0007 --l2_emb 0.0002 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.5 --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.02 --friend_prior_topk 2
run_exp phl_agran_h96_l50_b2_kl05_fp003 $PHL --n_epochs 60 --hidden_units 96 --maxlen 50 --num_blocks 2 --num_heads 3 --dropout_rate 0.40 --kl_reg 0.5 --lr 0.0007 --l2_emb 0.0002 --lr_scheduler cosine --time_span 256 --dis_span 384 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.20 --rel_bias_weight 1.0 --friend_prior_weight 0.03 --friend_prior_topk 2
run_exp phl_agran_h64_l75_b2_kl02_fp003 $PHL --n_epochs 60 --hidden_units 64 --maxlen 75 --num_blocks 2 --num_heads 2 --dropout_rate 0.40 --kl_reg 0.2 --lr 0.0008 --l2_emb 0.0002 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.5 --time_span 384 --dis_span 384 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.03 --friend_prior_topk 3
run_exp phl_agran_h64_l50_b4_kl01_nofriend $PHL --n_epochs 60 --hidden_units 64 --maxlen 50 --num_blocks 4 --num_heads 2 --dropout_rate 0.40 --kl_reg 0.1 --lr 0.001 --l2_emb 0.0001 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.5 --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.2 --time_score_weight 0.15 --rel_bias_weight 1.0 --no_friend_prior
run_exp phl_agran_h96_l50_b2_kl10_directed $PHL --n_epochs 55 --hidden_units 96 --maxlen 50 --num_blocks 2 --num_heads 3 --dropout_rate 0.45 --kl_reg 1.0 --lr 0.0005 --l2_emb 0.0002 --lr_scheduler step --lr_step_size 20 --lr_gamma 0.5 --time_span 256 --dis_span 512 --num_time_bins 512 --rel_score_weight 1.0 --time_score_weight 0.20 --rel_bias_weight 1.2 --directed_prior --friend_prior_weight 0.02 --friend_prior_topk 2
run_exp phl_agran_h128_l40_b2_kl02_fp002_bs192 $COMMON --dataset Yelp-PHL --batch_size 192 --eval_batch_size 384 --evaluate_every 5 --patience 30 --n_epochs 50 --hidden_units 128 --maxlen 40 --num_blocks 2 --num_heads 4 --dropout_rate 0.45 --kl_reg 0.2 --lr 0.0004 --l2_emb 0.0002 --lr_scheduler cosine --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.02 --friend_prior_topk 2

run_exp tpa_agran_h64_l50_b3_kl02_fp003 $TPA --n_epochs 90 --hidden_units 64 --maxlen 50 --num_blocks 3 --num_heads 2 --dropout_rate 0.30 --kl_reg 0.2 --lr 0.001 --l2_emb 0.0001 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.03 --friend_prior_topk 2
run_exp tpa_agran_h96_l50_b3_kl02_fp003 $TPA --n_epochs 80 --hidden_units 96 --maxlen 50 --num_blocks 3 --num_heads 3 --dropout_rate 0.35 --kl_reg 0.2 --lr 0.0007 --l2_emb 0.0001 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.03 --friend_prior_topk 2
run_exp tpa_agran_h128_l50_b2_kl01_fp003_bs256 $COMMON --dataset Yelp-TPA --batch_size 256 --eval_batch_size 512 --evaluate_every 5 --patience 35 --n_epochs 70 --hidden_units 128 --maxlen 50 --num_blocks 2 --num_heads 4 --dropout_rate 0.35 --kl_reg 0.1 --lr 0.0005 --l2_emb 0.0001 --lr_scheduler cosine --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.03 --friend_prior_topk 2
run_exp tpa_agran_h64_l75_b3_kl02_fp005 $TPA --n_epochs 80 --hidden_units 64 --maxlen 75 --num_blocks 3 --num_heads 2 --dropout_rate 0.35 --kl_reg 0.2 --lr 0.0008 --l2_emb 0.0002 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --time_span 384 --dis_span 384 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.20 --rel_bias_weight 1.0 --friend_prior_weight 0.05 --friend_prior_topk 3
run_exp tpa_agran_h96_l75_b2_kl05_fp005 $TPA --n_epochs 70 --hidden_units 96 --maxlen 75 --num_blocks 2 --num_heads 3 --dropout_rate 0.40 --kl_reg 0.5 --lr 0.0007 --l2_emb 0.0002 --lr_scheduler cosine --time_span 384 --dis_span 384 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.20 --rel_bias_weight 1.0 --friend_prior_weight 0.05 --friend_prior_topk 3
run_exp tpa_agran_h64_l100_b2_kl02_fp003 $TPA --n_epochs 70 --hidden_units 64 --maxlen 100 --num_blocks 2 --num_heads 2 --dropout_rate 0.40 --kl_reg 0.2 --lr 0.0008 --l2_emb 0.0002 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --time_span 512 --dis_span 384 --num_time_bins 256 --rel_score_weight 1.0 --time_score_weight 0.15 --rel_bias_weight 1.0 --friend_prior_weight 0.03 --friend_prior_topk 3
run_exp tpa_agran_h64_l50_b4_kl01_nofriend $TPA --n_epochs 80 --hidden_units 64 --maxlen 50 --num_blocks 4 --num_heads 2 --dropout_rate 0.35 --kl_reg 0.1 --lr 0.001 --l2_emb 0.0001 --lr_scheduler step --lr_step_size 30 --lr_gamma 0.5 --time_span 256 --dis_span 256 --num_time_bins 256 --rel_score_weight 1.2 --time_score_weight 0.15 --rel_bias_weight 1.0 --no_friend_prior
run_exp tpa_agran_h96_l50_b2_kl10_directed $TPA --n_epochs 70 --hidden_units 96 --maxlen 50 --num_blocks 2 --num_heads 3 --dropout_rate 0.40 --kl_reg 1.0 --lr 0.0005 --l2_emb 0.0002 --lr_scheduler step --lr_step_size 25 --lr_gamma 0.5 --time_span 256 --dis_span 512 --num_time_bins 512 --rel_score_weight 1.0 --time_score_weight 0.20 --rel_bias_weight 1.2 --directed_prior --friend_prior_weight 0.03 --friend_prior_topk 2
