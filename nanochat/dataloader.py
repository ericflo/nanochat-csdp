from collections import deque
import random

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

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

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
    batches = document_batches()

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

    # Get tokenizer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # CSDP special tokens for clear boundary marking
    csdp_start_id = tokenizer.encode_special("<|csdp_start|>")
    csdp_end_id = tokenizer.encode_special("<|csdp_end|>")

    # DDP setup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # Document batch generator
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx
        while True:
            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size
                pq_idx += 1
    batches = document_batches()

    # Token and weight buffers
    needed_tokens = B * T + 1
    token_buffer = deque()
    weight_buffer = deque()  # Parallel buffer for loss weights
    use_cuda_optimizations = device == "cuda"

    # Step counter for CSDP stage detection
    step_counter = resume_state_dict.get("csdp_step", 0) if resume_state_dict else 0
    last_csdp_text = ""
    last_csdp_stage = ""
    last_csdp_token_count = 0

    while True:
        # Accumulate enough tokens for one batch
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)

            # Process each document
            for doc_text in doc_batch:
                # Tokenize document with BOS
                doc_tokens = tokenizer.encode(doc_text, prepend=bos_token)

                # Determine if we should include CSDP for this document
                include_csdp = (
                    curriculum != "none" and
                    random.random() <= get_csdp_probability(step_counter, total_steps)
                )

                if include_csdp:
                    # Get CSDP content
                    domain = classify_domain(doc_text) if use_domain else None
                    csdp_text = get_csdp_block(
                        step=step_counter,
                        total_steps=total_steps,
                        curriculum=curriculum,
                        domain=domain,
                        include_graduation=enable_graduation
                    )
                    csdp_stage = get_stage(step_counter, total_steps)

                    # Tokenize CSDP
                    csdp_tokens = tokenizer.encode(csdp_text)

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

                    token_buffer.extend(combined_tokens)
                    weight_buffer.extend(combined_weights)
                else:
                    # No CSDP - just regular tokens with full weight
                    token_buffer.extend(doc_tokens)
                    weight_buffer.extend([1.0] * len(doc_tokens))

        # Build batch tensors
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        weights = [weight_buffer.popleft() for _ in range(needed_tokens)]

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

        csdp_info = {
            "step": step_counter,
            "stage": last_csdp_stage,
            "csdp_text": last_csdp_text,
            "csdp_token_count": last_csdp_token_count,
            "curriculum": curriculum,
            "loss_weight": loss_weight,
        }

        step_counter += 1

        yield inputs, targets, loss_weights, state_dict, csdp_info


def csdp_tokenizing_data_loader(B, T, split, csdp_config, **kwargs):
    """Helper that emits just (inputs, targets, loss_weights) without state/info."""
    for inputs, targets, loss_weights, state_dict, csdp_info in csdp_tokenizing_data_loader_with_state(
        B, T, split, csdp_config, **kwargs
    ):
        yield inputs, targets, loss_weights
