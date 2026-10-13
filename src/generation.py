"""Batch generation and per-token log-prob extraction (hand-written, no vLLM).

Decoder-only models require **left padding** for correct batch generation; the
response span of each row is tracked explicitly so that log-probs cover exactly
the sampled tokens (pads / early-stop fill excluded, eos kept).

Memory safety on 15 GB cards: generation runs in chunks of ``gen_chunk_size``
prompts, and log-probs are computed with a row-chunked forward + per-row
``cross_entropy`` so a full ``(B, L, |V|)`` logits tensor (≈15 GB for 64×768
tokens on Qwen2.5-1.5B) is never materialized.
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast


@torch.no_grad()
def batch_generate(model, tokenizer, prompts, max_new_tokens=256, temperature=1.0,
                   device="cuda", max_prompt_tokens=512, gen_chunk_size=16):
    """Generate a batch of responses from chat-formatted prompts.

    Args:
        model: causal LM (policy).
        tokenizer: tokenizer owning ``apply_chat_template`` (left padding).
        prompts: list of message lists (``[{"role": ..., "content": ...}]``).
        max_new_tokens: generation budget for the completion.
        temperature: generation temperature.
        device: device string.
        max_prompt_tokens: prompt truncation limit.
        gen_chunk_size: prompts generated per forward batch (memory cap).

    Returns:
        responses: list of decoded completion strings.
        full_ids: (batch, seq_len) prompt + completion tokens (left-padded,
            right-padded to a common width with ``pad_token_id``).
        response_spans: list of (start, end) token positions per row — the
            exactly-sampled span used for log-prob extraction.
        completion_lengths: list of response token counts.
        truncated: list of bool, True when a completion hit ``max_new_tokens``.
    """
    texts = []
    for prompt in prompts:
        if isinstance(prompt, str):
            texts.append(prompt)
        else:
            texts.append(tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True))

    pad_id = tokenizer.pad_token_id
    responses, response_spans = [], []
    completion_lengths, truncated = [], []
    chunk_full_ids = []

    for c0 in range(0, len(texts), gen_chunk_size):
        chunk_texts = texts[c0:c0 + gen_chunk_size]
        encodings = tokenizer(
            chunk_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        prompt_len = int(input_ids.size(1))  # uniform padded width (left pad)

        full_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            pad_token_id=pad_id,
        )

        seq_len = int(full_ids.size(1))
        for i in range(len(chunk_texts)):
            region = full_ids[i, prompt_len:]
            pad_positions = (region == pad_id).nonzero(as_tuple=True)[0]
            if pad_positions.numel() > 0:
                # stopped early (eos sampled, then pad fill): span ends before pads
                end = prompt_len + int(pad_positions[0])
                trunc = False
            else:
                end = seq_len
                trunc = True
            response_spans.append((prompt_len, end))
            completion_lengths.append(end - prompt_len)
            truncated.append(trunc)
            responses.append(tokenizer.decode(
                full_ids[i, prompt_len:end], skip_special_tokens=True))
        chunk_full_ids.append(full_ids)

    # align chunks to a common width by right-padding (mask/log-prob logic
    # treats pad ids as padding, so the fill never enters any response span)
    max_len = max(f.size(1) for f in chunk_full_ids)
    if len(chunk_full_ids) == 1 and chunk_full_ids[0].size(1) == max_len:
        full_ids = chunk_full_ids[0]
    else:
        full_ids = torch.full((len(texts), max_len), pad_id,
                              dtype=chunk_full_ids[0].dtype, device=device)
        row = 0
        for f in chunk_full_ids:
            full_ids[row:row + f.size(0), :f.size(1)] = f
            row += f.size(0)

    return responses, full_ids, response_spans, completion_lengths, truncated


def compute_log_probs_for_tokens(model, full_ids, response_spans,
                                 attention_mask=None, use_amp=False,
                                 device="cuda", row_chunk_size=4):
    """Per-token log-probs of each row's response span.

    Args:
        response_spans: list of (start, end) positions from ``batch_generate``.
        row_chunk_size: rows forwarded per chunk. The full-batch forward would
            materialize ``(B, L, |V|)`` logits (≈15 GB for 64×768 on a 1.5B
            model), so rows are processed in chunks and logits are dropped as
            soon as each row's span is gathered.

    Returns:
        List of 1-D float tensors (one per sample, variable length).

    Gradient note: with grad enabled (policy updates), each chunk keeps its own
    autograd graph sharing the same parameters; backward accumulates across
    chunks, so chunking is exact, not an approximation. Per-row
    ``cross_entropy`` computes the same log-probs as
    ``log_softmax(...).gather(...)`` without keeping an fp32 log-probs copy of
    the whole tensor alive until backward.
    """
    full_ids = full_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    device_type = "cuda" if "cuda" in str(device) else (
        "mps" if "mps" in str(device) else "cpu")
    n = full_ids.size(0)
    log_probs_list = [None] * n

    for c0 in range(0, n, row_chunk_size):
        c1 = min(c0 + row_chunk_size, n)
        with autocast(device_type=device_type, enabled=use_amp):
            logits = model(
                input_ids=full_ids[c0:c1],
                attention_mask=(attention_mask[c0:c1]
                                if attention_mask is not None else None),
            ).logits

        for i in range(c0, c1):
            start, end = response_spans[i]
            if end <= start or start < 1:
                log_probs_list[i] = torch.tensor([], device=device)
                continue
            row = i - c0
            response_token_ids = full_ids[i, start:end]
            # logits at position t predict token t+1 -> shift by one
            pred_logits = logits[row, start - 1:end - 1, :]
            # -CE(x, target) == log_softmax(x)[target]; computed outside the
            # autocast region on the bf16 logits so no fp32 (span, |V|) copy
            # is retained for backward
            log_probs_list[i] = -F.cross_entropy(
                pred_logits, response_token_ids, reduction="none")
        del logits

    return log_probs_list
