"""
CSDP (Contextual Scaffolding During Pretraining) Logging Module.

Provides comprehensive logging for CSDP experiments, including:
- Per-batch CSDP content and metadata
- Training metrics with CSDP-specific info
- Model samples at different training stages
- Attention pattern snapshots (optional)
- Run-level aggregation and analysis

This is designed for maximum-detail logging to support publication-quality
analysis of the CSDP experiment results.
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict


@dataclass
class BatchLogEntry:
    """Log entry for a single training batch."""
    step: int
    timestamp: float
    stage: str
    curriculum: str
    csdp_text: str  # Truncated for storage
    csdp_token_count: int
    domain: Optional[str]
    loss_weight: float
    training_loss: float
    csdp_loss: Optional[float]  # Loss on CSDP tokens only (if computed)


@dataclass
class SampleLogEntry:
    """Log entry for model samples."""
    step: int
    timestamp: float
    stage: str
    prompt: str
    completion: str
    temperature: float
    prompt_type: str  # "self_knowledge", "factual", "creative", etc.


@dataclass
class MetricsLogEntry:
    """Log entry for training metrics."""
    step: int
    timestamp: float
    stage: str
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    tokens_per_sec: int
    mfu: float


class CSDPLogger:
    """
    Comprehensive logging system for CSDP experiments.

    Creates a structured directory hierarchy for each run:
    {run_dir}/
        batch_logs/
            batch_details.jsonl   # Every batch
        samples/
            step_000000.json      # Model samples at each stage
        metrics/
            training_metrics.jsonl
        attention/                # Optional attention patterns
        summary.json              # Run summary
    """

    def __init__(self, run_dir: str, curriculum: str, run_id: str):
        """
        Initialize the CSDP logger.

        Args:
            run_dir: Base directory for this run's logs
            curriculum: Curriculum name (aria/sage/nova/heart/bare)
            run_id: Unique identifier for this run
        """
        self.run_dir = run_dir
        self.curriculum = curriculum
        self.run_id = run_id
        self.start_time = time.time()

        # Create directory structure
        self.dirs = {
            'batches': os.path.join(run_dir, 'batch_logs'),
            'samples': os.path.join(run_dir, 'samples'),
            'metrics': os.path.join(run_dir, 'metrics'),
            'attention': os.path.join(run_dir, 'attention_patterns'),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        # Initialize log files
        self.batch_log_path = os.path.join(self.dirs['batches'], 'batch_details.jsonl')
        self.metrics_log_path = os.path.join(self.dirs['metrics'], 'training_metrics.jsonl')

        # In-memory tracking for efficiency
        self.stage_transition_steps: Dict[str, int] = {}
        self.stage_sample_counts: Dict[str, int] = defaultdict(int)
        self.total_batches_logged = 0
        self.last_stage = ""

        # Write run metadata
        self._write_run_metadata()

    def _write_run_metadata(self):
        """Write initial run metadata."""
        metadata = {
            "run_id": self.run_id,
            "curriculum": self.curriculum,
            "start_time": self.start_time,
            "start_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
        }
        metadata_path = os.path.join(self.run_dir, "run_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_batch(self, step: int, batch_info: Dict[str, Any], log_every: int = 1):
        """
        Log a training batch with CSDP info.

        Args:
            step: Current training step
            batch_info: Dict with keys:
                - stage: Current CSDP stage
                - csdp_text: CSDP content (will be truncated)
                - csdp_token_count: Number of CSDP tokens
                - domain: Optional domain classification
                - loss_weight: CSDP loss weight
                - training_loss: Total training loss
                - csdp_loss: Optional loss on CSDP tokens only
            log_every: Log every N batches (for performance)
        """
        # Track stage transitions
        current_stage = batch_info.get('stage', '')
        if current_stage and current_stage != self.last_stage:
            self.stage_transition_steps[current_stage] = step
            self.last_stage = current_stage

        # Log every N batches
        if step % log_every != 0:
            return

        entry = BatchLogEntry(
            step=step,
            timestamp=time.time(),
            stage=batch_info.get('stage', ''),
            curriculum=self.curriculum,
            csdp_text=batch_info.get('csdp_text', '')[:500],  # Truncate
            csdp_token_count=batch_info.get('csdp_token_count', 0),
            domain=batch_info.get('domain'),
            loss_weight=batch_info.get('loss_weight', 0.0),
            training_loss=batch_info.get('training_loss', 0.0),
            csdp_loss=batch_info.get('csdp_loss'),
        )

        with open(self.batch_log_path, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

        self.total_batches_logged += 1

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Log training metrics.

        Args:
            step: Current training step
            metrics: Dict with keys:
                - stage: Current CSDP stage
                - train_loss: Training loss
                - val_loss: Validation loss (optional)
                - learning_rate: Current learning rate
                - tokens_per_sec: Training throughput
                - mfu: Model FLOP utilization
        """
        entry = MetricsLogEntry(
            step=step,
            timestamp=time.time(),
            stage=metrics.get('stage', ''),
            train_loss=metrics.get('train_loss', 0.0),
            val_loss=metrics.get('val_loss'),
            learning_rate=metrics.get('learning_rate', 0.0),
            tokens_per_sec=metrics.get('tokens_per_sec', 0),
            mfu=metrics.get('mfu', 0.0),
        )

        with open(self.metrics_log_path, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

    def log_sample(self, step: int, prompt: str, completion: str,
                   stage: str, temperature: float = 0.0,
                   prompt_type: str = "general"):
        """
        Log a model sample.

        Args:
            step: Current training step
            prompt: Input prompt
            completion: Model completion
            stage: Current CSDP stage
            temperature: Sampling temperature
            prompt_type: Type of prompt for categorization
        """
        entry = SampleLogEntry(
            step=step,
            timestamp=time.time(),
            stage=stage,
            prompt=prompt,
            completion=completion,
            temperature=temperature,
            prompt_type=prompt_type,
        )

        sample_file = os.path.join(self.dirs['samples'], f'step_{step:06d}.json')

        # Append to existing file if present (multiple samples per step)
        samples = []
        if os.path.exists(sample_file):
            with open(sample_file, 'r') as f:
                samples = json.load(f)

        samples.append(asdict(entry))

        with open(sample_file, 'w') as f:
            json.dump(samples, f, indent=2)

        self.stage_sample_counts[stage] += 1

    def log_attention_snapshot(self, step: int, attention_weights: Any,
                               csdp_positions: List[int], layer_idx: int = 0):
        """
        Log attention patterns to CSDP tokens.

        This is expensive, so should be called sparingly (e.g., every 1000 steps).

        Args:
            step: Current training step
            attention_weights: Attention tensor (will be converted to list)
            csdp_positions: Token positions that are CSDP content
            layer_idx: Which layer's attention to log
        """
        try:
            import torch
            if isinstance(attention_weights, torch.Tensor):
                # Extract attention to CSDP positions
                # attention_weights: (batch, heads, seq, seq)
                attn_to_csdp = attention_weights[0, :, :, csdp_positions].mean(dim=0).cpu().numpy()
                attention_data = attn_to_csdp.tolist()
            else:
                attention_data = attention_weights
        except Exception as e:
            attention_data = {"error": str(e)}

        snapshot = {
            "step": step,
            "timestamp": time.time(),
            "layer": layer_idx,
            "csdp_positions": csdp_positions,
            "attention_to_csdp": attention_data,
        }

        attn_file = os.path.join(self.dirs['attention'], f'step_{step:06d}_layer_{layer_idx}.json')
        with open(attn_file, 'w') as f:
            json.dump(snapshot, f)

    def generate_run_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the run.

        Returns:
            Dict with run statistics and metrics
        """
        end_time = time.time()

        summary = {
            "run_id": self.run_id,
            "curriculum": self.curriculum,
            "start_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "end_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            "duration_seconds": end_time - self.start_time,
            "duration_minutes": (end_time - self.start_time) / 60,
            "total_batches_logged": self.total_batches_logged,
            "stage_transitions": self.stage_transition_steps,
            "samples_per_stage": dict(self.stage_sample_counts),
        }

        # Write summary file
        summary_path = os.path.join(self.run_dir, "run_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def load_batch_logs(self, max_entries: int = -1) -> List[Dict]:
        """Load batch logs from file."""
        entries = []
        if os.path.exists(self.batch_log_path):
            with open(self.batch_log_path, 'r') as f:
                for i, line in enumerate(f):
                    if max_entries > 0 and i >= max_entries:
                        break
                    entries.append(json.loads(line))
        return entries

    def load_metrics_logs(self) -> List[Dict]:
        """Load metrics logs from file."""
        entries = []
        if os.path.exists(self.metrics_log_path):
            with open(self.metrics_log_path, 'r') as f:
                for line in f:
                    entries.append(json.loads(line))
        return entries

    def load_samples(self, step: Optional[int] = None) -> List[Dict]:
        """Load samples from file(s)."""
        samples = []
        if step is not None:
            sample_file = os.path.join(self.dirs['samples'], f'step_{step:06d}.json')
            if os.path.exists(sample_file):
                with open(sample_file, 'r') as f:
                    samples = json.load(f)
        else:
            # Load all samples
            for fname in sorted(os.listdir(self.dirs['samples'])):
                if fname.endswith('.json'):
                    with open(os.path.join(self.dirs['samples'], fname), 'r') as f:
                        samples.extend(json.load(f))
        return samples


def get_csdp_logger(curriculum: str, run_name: str) -> Optional[CSDPLogger]:
    """
    Factory function to create a CSDP logger.

    Only creates logger if curriculum is not "none".

    Args:
        curriculum: Curriculum name
        run_name: Run identifier

    Returns:
        CSDPLogger instance or None if curriculum is "none"
    """
    if curriculum == "none":
        return None

    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    run_dir = os.path.join(base_dir, "csdp_runs", f"{curriculum}_{run_name}")

    return CSDPLogger(run_dir, curriculum, run_name)


class CSDPLoggerStub:
    """
    No-op stub for when CSDP logging is disabled.

    All methods are no-ops, so code can call logger methods without
    checking if logging is enabled.
    """

    def log_batch(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_sample(self, *args, **kwargs):
        pass

    def log_attention_snapshot(self, *args, **kwargs):
        pass

    def generate_run_summary(self) -> Dict:
        return {}
