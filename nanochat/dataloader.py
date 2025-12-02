from collections import deque
import logging
import random

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


def _create_document_batch_generator(split, tokenizer_batch_size, resume_state_dict=None):
    """
    Create a generator that yields document batches from parquet files.

    This is shared between the standard and CSDP dataloaders to avoid code duplication.

    Args:
        split: 'train' or 'val'
        tokenizer_batch_size: Batch size for yielding documents
        resume_state_dict: Optional state for resuming

    Yields:
        (doc_batch, (pq_idx, rg_idx)) tuples
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    pq_idx = resume_pq_idx

    while True:  # Iterate infinitely (multi-epoch)
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if resume_rg_idx is not None:
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # Advance by 1 so we don't repeat data
                rg_idx = base_idx * ddp_world_size + ddp_rank
                resume_rg_idx = None  # Only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                rg_idx += ddp_world_size
            pq_idx += 1


def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # Use shared document batch generator
    batches = _create_document_batch_generator(split, tokenizer_batch_size, resume_state_dict)

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


def csdp_tokenizing_data_loader_with_state(
    B, T, split,
    csdp_config,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None
):
    """
    Stream pretraining text with CSDP (Contextual Scaffolding During Pretraining).

    This is a CSDP-enabled variant of tokenizing_distributed_data_loader_with_state.
    It injects curriculum-based context at the start of each document and returns
    loss weights for weighted training.

    Args:
        B: Batch size
        T: Sequence length
        split: 'train' or 'val'
        csdp_config: CSDPConfig object with curriculum, loss_weight, etc.
        tokenizer_threads: Threads for tokenization
        tokenizer_batch_size: Batch size for tokenization
        device: Target device
        resume_state_dict: Optional state for resuming

    Yields:
        inputs: (B, T) input token ids
        targets: (B, T) target token ids
        loss_weights: (B, T) per-token loss weights (csdp_loss_weight for CSDP, 1.0 for training)
        state_dict: For checkpointing/resuming
        csdp_info: Dict with logging info (step, stage, csdp_text sample)
    """
    from nanochat.csdp import (
        get_csdp_block, get_csdp_probability, get_stage, classify_domain
    )

    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # CSDP config
    curriculum = csdp_config.curriculum if csdp_config else "none"
    loss_weight = csdp_config.loss_weight if csdp_config else 1.0
    use_domain = csdp_config.use_domain_context if csdp_config else False
    enable_graduation = csdp_config.enable_graduation if csdp_config else False
    total_steps = csdp_config.total_steps if csdp_config else 0
    max_csdp_ratio = csdp_config.max_csdp_ratio if csdp_config else 0.15

    # Get tokenizer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # CSDP special tokens for clear boundary marking
    csdp_start_id = tokenizer.encode_special("<|csdp_start|>")
    csdp_end_id = tokenizer.encode_special("<|csdp_end|>")

    # Use shared document batch generator (refactored to avoid code duplication)
    batches = _create_document_batch_generator(split, tokenizer_batch_size, resume_state_dict)

    # Combined token-weight buffer (single buffer prevents desync)
    # Each element is a (token_id, loss_weight) tuple
    needed_tokens = B * T + 1
    token_weight_buffer = deque()  # (token, weight) tuples - keeps buffers synchronized
    use_cuda_optimizations = device == "cuda"

    # Create a seeded RNG for reproducibility if configured
    csdp_rng = csdp_config.create_rng() if csdp_config else None

    # Step counter for CSDP stage detection
    step_counter = resume_state_dict.get("csdp_step", 0) if resume_state_dict else 0
    last_csdp_text = ""
    last_csdp_stage = ""
    last_csdp_token_count = 0

    # CSDP injection rate tracking for experiment analysis
    docs_total = 0
    docs_with_csdp = 0
    docs_skipped_ratio = 0  # Skipped due to max_csdp_ratio limit
    docs_skipped_annealing = 0  # Skipped due to graduation annealing
    log_interval = 1000  # Log stats every N documents

    while True:
        # Accumulate enough tokens for one batch
        while len(token_weight_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)

            # Process each document
            for doc_text in doc_batch:
                docs_total += 1

                # Tokenize document with BOS
                doc_tokens = tokenizer.encode(doc_text, prepend=bos_token)

                # Determine if we should include CSDP for this document
                # Use seeded RNG if available for reproducibility
                rand_val = csdp_rng.random() if csdp_rng else random.random()
                csdp_prob = get_csdp_probability(step_counter, total_steps)
                include_csdp = (
                    curriculum != "none" and
                    rand_val <= csdp_prob
                )

                # Track if annealing caused skip
                if curriculum != "none" and rand_val > csdp_prob:
                    docs_skipped_annealing += 1

                if include_csdp:
                    # Get CSDP content
                    domain = classify_domain(doc_text) if use_domain else None
                    csdp_text = get_csdp_block(
                        step=step_counter,
                        total_steps=total_steps,
                        curriculum=curriculum,
                        domain=domain,
                        include_graduation=enable_graduation,
                        rng=csdp_rng
                    )
                    csdp_stage = get_stage(step_counter, total_steps)

                    # Tokenize CSDP
                    csdp_tokens = tokenizer.encode(csdp_text)

                    # Check if CSDP would dominate the document (exceed max_csdp_ratio)
                    # CSDP section includes: csdp_start + csdp_tokens + csdp_end
                    csdp_section_len = 1 + len(csdp_tokens) + 1
                    total_len = len(doc_tokens) + csdp_section_len
                    csdp_ratio = csdp_section_len / total_len if total_len > 0 else 0

                    if csdp_ratio > max_csdp_ratio:
                        # CSDP would dominate this short document, skip it
                        # This prevents very long curricula (like HEART's full_comprehension)
                        # from overwhelming short documents
                        docs_skipped_ratio += 1
                        token_weight_buffer.extend((tok, 1.0) for tok in doc_tokens)
                        continue

                    # Successfully injecting CSDP
                    docs_with_csdp += 1

                    # Build sequence: [BOS] + [<|csdp_start|>] + [CSDP content] + [<|csdp_end|>] + [rest of doc]
                    # This provides clear token-level boundaries for:
                    # 1. Analysis of attention patterns to CSDP
                    # 2. Precise loss weighting boundary
                    # 3. Model learning to recognize context switch
                    combined_tokens = (
                        [doc_tokens[0]] +  # BOS
                        [csdp_start_id] +  # CSDP start marker
                        csdp_tokens +
                        [csdp_end_id] +    # CSDP end marker - transition to training
                        doc_tokens[1:]     # Rest of document (without BOS)
                    )

                    # Build weight sequence - CSDP tokens get reduced weight
                    # Note: csdp_end_id marks the transition, so it gets CSDP weight
                    csdp_section_len = 1 + len(csdp_tokens) + 1  # start + content + end
                    combined_weights = (
                        [loss_weight] +                       # BOS
                        [loss_weight] +                       # csdp_start
                        [loss_weight] * len(csdp_tokens) +    # CSDP content
                        [loss_weight] +                       # csdp_end
                        [1.0] * (len(doc_tokens) - 1)         # Full weight for document
                    )

                    # Store for logging
                    last_csdp_text = csdp_text[:500]  # Truncate for logging
                    last_csdp_stage = csdp_stage
                    last_csdp_token_count = csdp_section_len  # For logging

                    # Add token-weight pairs to buffer (prevents desync)
                    token_weight_buffer.extend(zip(combined_tokens, combined_weights))
                else:
                    # No CSDP - just regular tokens with full weight
                    token_weight_buffer.extend((tok, 1.0) for tok in doc_tokens)

        # Build batch tensors - extract from combined buffer
        token_weight_pairs = [token_weight_buffer.popleft() for _ in range(needed_tokens)]
        tokens = [pair[0] for pair in token_weight_pairs]
        weights = [pair[1] for pair in token_weight_pairs]

        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        weight_scratch = torch.tensor(weights, dtype=torch.float32, pin_memory=use_cuda_optimizations)

        # Create inputs/targets (1D tensors)
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        weights_cpu = weight_scratch[1:]  # Weights align with targets

        # Reshape to 2D and move to GPU
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        loss_weights = weights_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        state_dict = {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "csdp_step": step_counter
        }

        # Calculate injection rate stats
        injection_rate = docs_with_csdp / docs_total if docs_total > 0 else 0.0
        skip_ratio_rate = docs_skipped_ratio / docs_total if docs_total > 0 else 0.0
        skip_anneal_rate = docs_skipped_annealing / docs_total if docs_total > 0 else 0.0

        csdp_info = {
            "step": step_counter,
            "stage": last_csdp_stage,
            "csdp_text": last_csdp_text,
            "csdp_token_count": last_csdp_token_count,
            "curriculum": curriculum,
            "loss_weight": loss_weight,
            # Injection rate stats for experiment analysis
            "docs_total": docs_total,
            "docs_with_csdp": docs_with_csdp,
            "docs_skipped_ratio": docs_skipped_ratio,
            "docs_skipped_annealing": docs_skipped_annealing,
            "injection_rate": injection_rate,
        }

        # Periodic logging of injection rate stats
        if docs_total > 0 and docs_total % log_interval == 0:
            logger.info(
                f"CSDP injection stats (last {log_interval} docs): "
                f"injected={injection_rate:.1%}, "
                f"skipped_ratio={skip_ratio_rate:.1%}, "
                f"skipped_annealing={skip_anneal_rate:.1%}, "
                f"stage={last_csdp_stage}"
            )

        step_counter += 1

        yield inputs, targets, loss_weights, state_dict, csdp_info


def csdp_tokenizing_data_loader(B, T, split, csdp_config, **kwargs):
    """Helper that emits just (inputs, targets, loss_weights) without state/info."""
    for inputs, targets, loss_weights, state_dict, csdp_info in csdp_tokenizing_data_loader_with_state(
        B, T, split, csdp_config, **kwargs
    ):
        yield inputs, targets, loss_weights
