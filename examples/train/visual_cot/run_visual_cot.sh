#!/bin/bash
# Run all in terminal
export DATA_DIR="$HOME/data/visual_cot"
export HYDRA_FULL_ERROR=1
export NUM_GPUS=1
export _SKYRL_USE_NEW_INFERENCE=1
export CUDA_VISIBLE_DEVICES=4
export RAY_TMPDIR="/tmp/ray_$USER"
mkdir -p "$RAY_TMPDIR"
source .venv/bin/activate

uv sync --extra fsdp
ps -u $USER -o pid= -o command= | grep '[r]ay' | awk '{print $1}' | xargs -r kill -9
ps -u $USER -o pid= -o command= | grep '[d]ebugpy' | awk '{print $1}' | xargs -r kill -9
rm -rf /tmp/ray_$USER/session_*
python examples/train/visual_cot/visual_cot_dataset.py --max_dataset_length 16

python examples/train/visual_cot/main_visual_cot.py \
      data.train_data="['$DATA_DIR/train.parquet']" \
      data.val_data="['$DATA_DIR/test.parquet']" \
      trainer.policy.model.path="Qwen/Qwen3-VL-2B-Instruct" \
      trainer.ref.model.path="Qwen/Qwen3-VL-2B-Instruct" \
      trainer.resume_mode=none \
      trainer.use_sample_packing=false \
      trainer.algorithm.advantage_estimator="grpo" \
      trainer.algorithm.use_kl_loss=true \
      trainer.algorithm.kl_loss_coef=0.001 \
      trainer.eval_before_train=false \
      trainer.epochs=3 \
      trainer.train_batch_size=16 \
      trainer.policy_mini_batch_size=16 \
      trainer.micro_forward_batch_size_per_gpu=16 \
      trainer.micro_train_batch_size_per_gpu=16 \
      trainer.placement.colocate_all=true \
      trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
      trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
      trainer.strategy=fsdp2 \
      trainer.hf_save_interval=1 \
      trainer.export_path="$HOME/SkyRL/outputs/Qwen3-VL-2B-Instruct-visual-cot-grpo" \
      generator.backend=vllm \
      generator.batched=false \
      generator.use_conversation_multi_turn=true \
      generator.n_samples_per_prompt=2 \
      generator.inference_engine_tensor_parallel_size=1 \
      generator.inference_engine_data_parallel_size=1 \
      generator.num_inference_engines=1 \
      generator.sampling_params.max_generate_length=256 \
      generator.sampling_params.temperature=0.7 \
      environment.env_class=visual_cot \
      trainer.run_name="qwen3-vl-2b-visual-cot-grpo-1gpu" \
      trainer.project_name="skyrl-tutorial" \
      trainer.log_path="/tmp/skyrl-logs-suhas" > training.log 2>&1