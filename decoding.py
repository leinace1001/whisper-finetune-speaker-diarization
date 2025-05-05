import torch
import torch.nn.functional as F

from settings import *

def beam_search_decode(model, encoder_output, per_fix, eos_token, beam_size=3):
    """
    Performs beam search decoding on a Transformer-based ASR decoder.

    Args:
        model: Transformer-based ASR model with decoder.
        encoder_output: Output from encoder, shape: [batch, seq_len, d_model]
        eos_token: ID of end-of-sequence token.
        beam_size: Number of beams to explore.
        max_len: Maximum sequence length to decode.

    Returns:
        The best sequence predicted by beam search.
    """
    device = encoder_output.device

    # Initialize beams with start token
    beams = [(torch.tensor(per_fix, device=device), 0)]  # (sequence, score)

    for _ in range(DECODE_MAX_LENGTH-len(per_fix)):
        candidates = []

        for seq, score in beams:
            if seq[-1].item() == eos_token:
                # If EOS reached, add unchanged
                candidates.append((seq, score))
                continue

            # Predict next token probabilities
            tgt_input = seq.unsqueeze(0)  # [1, seq_len]
            logits = model.logits(tgt_input, encoder_output)  # [1, seq_len, vocab_size]

            logits = logits[0, -1, :]  # [vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)

            # Select top-k candidates
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for log_prob, idx in zip(topk_log_probs, topk_indices):
                new_seq = torch.cat([seq, idx.view(1)])
                candidates.append((new_seq, score + log_prob.item()))

        # Sort all candidates and pick the best beam_size sequences
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        # If all sequences end with EOS, terminate early
        if all(seq[-1].item() == eos_token for seq, _ in beams):
            break

    return beams

