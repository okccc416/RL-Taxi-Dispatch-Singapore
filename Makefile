# ============================================================================
# Makefile — End-to-End RL Taxi Dispatch Pipeline
# ============================================================================
# Usage:
#   make all          Run the full pipeline (data → train → evaluate)
#   make data         Generate road network, H3 mapping, tidal demand
#   make train        Train PPO-Ours (α=0.5) and PPO-Ablation (α=0.0)
#   make evaluate     Multi-seed evaluation with publication plots
#   make dashboard    Install & launch the real-time dispatch dashboard
#   make clean        Remove generated artefacts
# ============================================================================

PYTHON      ?= python
ITERATIONS  ?= 50
NUM_TAXIS   ?= 20
NUM_WORKERS ?= 2
CKPT_FREQ   ?= 10

.PHONY: all data train train-ours train-ablation evaluate dashboard clean \
       train-500 evaluate-500 scale-test

# ── Full pipeline ──────────────────────────────────────────────────────────
all: data train evaluate

# ── Step 1: Data Engineering ───────────────────────────────────────────────
data:
	@echo "════════════════════════════════════════════════════════"
	@echo "  STEP 1 — Data Pipeline (Road Network + H3 + Demand)"
	@echo "════════════════════════════════════════════════════════"
	$(PYTHON) data_pipeline.py

# ── Step 2: Training ──────────────────────────────────────────────────────
train: train-ours train-ablation

train-ours:
	@echo "════════════════════════════════════════════════════════"
	@echo "  STEP 2a — Train PPO-Ours (density α = 0.5)"
	@echo "════════════════════════════════════════════════════════"
	$(PYTHON) train_rllib.py \
		--iterations $(ITERATIONS) \
		--num-taxis $(NUM_TAXIS) \
		--num-workers $(NUM_WORKERS) \
		--checkpoint-freq $(CKPT_FREQ)

train-ablation:
	@echo "════════════════════════════════════════════════════════"
	@echo "  STEP 2b — Train PPO-Ablation (density α = 0.0)"
	@echo "════════════════════════════════════════════════════════"
	$(PYTHON) train_rllib.py \
		--iterations $(ITERATIONS) \
		--num-taxis $(NUM_TAXIS) \
		--num-workers $(NUM_WORKERS) \
		--checkpoint-freq $(CKPT_FREQ) \
		--ablation

# ── Step 3: Evaluation ────────────────────────────────────────────────────
evaluate:
	@echo "════════════════════════════════════════════════════════"
	@echo "  STEP 3 — Multi-Seed Evaluation & Publication Plots"
	@echo "════════════════════════════════════════════════════════"
	$(PYTHON) evaluate_and_plot.py \
		--ckpt-ours checkpoints_ours \
		--ckpt-ablation checkpoints_ablation

# ── Dashboard ─────────────────────────────────────────────────────────────
dashboard:
	@echo "════════════════════════════════════════════════════════"
	@echo "  Launching Real-Time Dispatch Dashboard"
	@echo "════════════════════════════════════════════════════════"
	cd dashboard; npm install; npm run dev

# ── Streaming demo (Kafka producer + Decision Gateway) ────────────────────
stream:
	@echo "Starting Decision Gateway on ws://localhost:8765 ..."
	$(PYTHON) decision_gateway.py --standalone --with-test-client

# ═══════════════════════════════════════════════════════════════════════════
# 500-TAXI INDUSTRIAL SCALE TEST  (outputs to *_500/ dirs, non-destructive)
# ═══════════════════════════════════════════════════════════════════════════
train-500:
	@echo "════════════════════════════════════════════════════════"
	@echo "  SCALE — Train PPO-Ours (500 taxis)"
	@echo "════════════════════════════════════════════════════════"
	$(PYTHON) train_rllib.py \
		--iterations $(ITERATIONS) \
		--num-taxis 500 \
		--num-workers $(NUM_WORKERS) \
		--checkpoint-freq $(CKPT_FREQ)
	@echo "════════════════════════════════════════════════════════"
	@echo "  SCALE — Train PPO-Ablation (500 taxis)"
	@echo "════════════════════════════════════════════════════════"
	$(PYTHON) train_rllib.py \
		--iterations $(ITERATIONS) \
		--num-taxis 500 \
		--num-workers $(NUM_WORKERS) \
		--checkpoint-freq $(CKPT_FREQ) \
		--ablation

evaluate-500:
	@echo "════════════════════════════════════════════════════════"
	@echo "  SCALE — Evaluate (500 taxis)"
	@echo "════════════════════════════════════════════════════════"
	$(PYTHON) evaluate_and_plot.py --num-taxis 500

scale-test: train-500 evaluate-500

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	rm -rf checkpoints_ours checkpoints_ablation
	rm -rf checkpoints_ours_* checkpoints_ablation_*
	rm -rf results results_* figures figures_*
	rm -rf __pycache__ .rllib_*
	rm -rf dashboard/.next dashboard/node_modules
