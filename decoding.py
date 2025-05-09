import torch
import torch.nn.functional as F
from whisper.decoding import *
from settings import *


class DecodingWithSpeakerLabels:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder

    def __init__(self, model, initial_tokens, beam_size, eos_token):
        self.model = model
        self.initial_tokens = initial_tokens
        self.inference = PyTorchInference(model, len(self.initial_tokens))
        self.decoder = BeamSearchDecoder(
                beam_size, eos_token, self.inference
            )
        self.beam_size = beam_size
        self.sequence_ranker = MaximumLikelihoodRanker(1)
        self.eos = eos_token
    
    def run(self, audio_features):
        tokens = torch.tensor(self.initial_tokens, device=audio_features.device).repeat(self.beam_size, 1)
        
        sum_logprobs: Tensor = torch.zeros(self.beam_size, device=audio_features.device)
        try:
            for _ in range(DECODE_MAX_LENGTH-len(self.initial_tokens)):
                logits = self.inference.logits(tokens, audio_features)
                logits = logits[:, -1]
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)
            
                if completed:
                    break
        finally:
            self.inference.cleanup_caching()
        tokens = tokens.reshape(1, self.beam_size, -1)
        sum_logprobs = sum_logprobs.reshape(1, self.beam_size)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[len(self.initial_tokens) : (t == self.eos).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        return tokens[0]

