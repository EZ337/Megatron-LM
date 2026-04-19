#!/bin/bash

# Smoke test for Mixtral 8x7B architecture on a single node with 8x L4 GPUs.
#
# Key changes from the full training script:
#   - NullTokenizer: no tokenizer file needed
#   - --mock-data: no dataset files needed; synthetic data is generated on the fly
#   - Model size heavily reduced: fewer layers, smaller hidden size, shorter seq length
#   - Parallelism adjusted for a single 8-GPU node (TP=2, PP=2, EP=2)
#   - train-iters set to 20 (roughly 5-10 min on 8x L4)
#   - No checkpoint saving or loading
#   - global-batch-size reduced to 16 to fit L4 VRAM (24GB each)
#
# Usage:
#   PYTHONPATH=/workspace/Megatron-LM:$PYTHONPATH bash smoke_test_mixtral_8x7b.sh
#
# Run from inside /workspace/Megatron-LM

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---------------------------------------------------------------------------
# Distributed setup -- single node only
# ---------------------------------------------------------------------------
GPUS_PER_NODE=4
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6001"}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ---------------------------------------------------------------------------
# Model architecture -- scaled-down Mixtral shape.
#
# Original Mixtral 8x7B values are shown in comments for reference.
# The overall structure (RMSNorm, RoPE, SwiGLU, GQA, MoE) is preserved so
# this tests the same code paths as the real model.
#
# --num-layers:        4   (was 32)   -- fewer transformer blocks
# --hidden-size:       1024 (was 4096) -- narrower residual stream
# --ffn-hidden-size:   3584 (was 14336) -- kept ratio ~3.5x hidden-size
# --num-attention-heads: 16 (was 32)  -- fewer heads, still divisible by num-query-groups
# --num-query-groups:  4   (was 8)    -- GQA key/value head count
# --seq-length:        512 (was 4096) -- shorter sequences to reduce memory
# --max-position-embeddings: 4096 (was 32768) -- must be >= seq-length
# ---------------------------------------------------------------------------
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 8
    --hidden-size 512
    --ffn-hidden-size 4096
    --num-attention-heads 4
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

# ---------------------------------------------------------------------------
# MoE settings -- kept identical to the original script.
# --num-experts 8 and --moe-router-topk 2 matches Mixtral 8x7B exactly.
# ---------------------------------------------------------------------------
MOE_ARGS=(
    --num-experts 4
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type stagedalltoall
    # --moe-enable-benchmark
    --overlap-param-gather
    --overlap-grad-reduce
)

# ---------------------------------------------------------------------------
# Data -- NullTokenizer + mock data, no files required.
#
# --vocab-size: 32000 matches Mixtral's original vocabulary size so the
# embedding table dimensions are realistic. If you later switch to a real
# tokenizer with a different vocab size, update this value.
# ---------------------------------------------------------------------------
DATA_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 32000
    --mock-data
)

# ---------------------------------------------------------------------------
# Training hyperparameters.
#
# --micro-batch-size 1 and --global-batch-size 16 keep peak VRAM low on L4s.
# --train-iters 20 is enough to confirm forward/backward pass + optimizer
# step work correctly across all GPUs. At ~15-30s per iter on 8x L4 this
# lands in the 5-10 minute window.
# ---------------------------------------------------------------------------
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1e-4
    --train-iters 10
    --lr-decay-iters 10
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 2
    --clip-grad 1.0
    --bf16
)

# ---------------------------------------------------------------------------
# Parallelism -- tuned for 8x L4 single node.
#
# TP=2: tensor parallel across 2 GPUs -- splits attention and MLP weight
#   matrices column/row-wise. Reduces per-GPU memory at the cost of
#   intra-node all-reduce communication.
#
# PP=2: pipeline parallel across 2 GPU stages -- splits the 4 layers into
#   2 stages of 2 layers each. With only 4 layers this is the practical min.
#
# EP=2: expert parallel across 2 GPUs -- each GPU holds 4 of the 8 experts.
#   Routing still happens globally but expert compute is local.
#
# TP x PP x EP = 2 x 2 x 2 = 8 GPUs total, which matches WORLD_SIZE.
#
# --sequence-parallel: activations in TP regions are sharded along the
#   sequence dimension, saving memory proportional to TP degree.
# ---------------------------------------------------------------------------
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 4
    --use-distributed-optimizer
    # --sequence-parallel
)

# ---------------------------------------------------------------------------
# Logging -- frequent output so you can watch progress, no saving.
# ---------------------------------------------------------------------------
LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 999999
    --eval-iters 0
    #--no-save
)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
