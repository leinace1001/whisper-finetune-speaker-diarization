import torch
import torch.nn as nn
from whisper.model import *
from whisper.decoding import DecodingTask
from transformers import Seq2SeqTrainer

from settings import *


class AudioEncoderWithSpeakers(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, max_spk_num:int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.speaker_label_embedder = nn.Embedding(1+2*max_spk_num, n_state, padding_idx=0)

    def forward(self, x: Tensor, speaker_labels:Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        batch_size, n_ctx, n_state = x.shape
        speaker_embedings = self.speaker_label_embedder(speaker_labels)
        
        #speaker_embedings = F.interpolate(speaker_embedings.unsqueeze(1), n_ctx, mode="nearest").squeeze(1)
        
        x = (x + self.positional_embedding + speaker_embedings).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
    
class WhisperWithSpeakers(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoderWithSpeakers(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            MAX_SPK_NUM
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        
    def forward(self, mel, speakers, tokens):
        latent = self.encoder(mel, speakers)

        return self.decoder(tokens, latent)
    
    def embed_audio(self, mel, speakers):
        return self.encoder(mel, speakers)
    
    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)
    
    

class MultimodalSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer to handle multimodal inputs (text + audio + speaker labels).
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function to handle multimodal data.

        Args:
            model: The transformer model
            inputs: Dictionary containing 'input_ids', 'attention_mask', 'audio_features', 'speaker_labels'
            return_outputs: Whether to return the model's outputs

        Returns:
            Loss tensor
        """

    
        # Speaker ID or one-hot encoded labels

        # Forward pass through the model
        
        outputs = model(
            inputs["mel"],
            inputs["speakers"],
            inputs["tokens"][:,:-1],
        )

        # Default loss from the model (e.g., cross-entropy for text)
        weights = torch.ones(outputs.shape[-1]).to(outputs.device)
        weights[15:25] = 2.
        weights[50257] = 0.5
        
        loss = F.cross_entropy(outputs.view(-1, outputs.shape[-1]), (inputs["tokens"][:,1:]).reshape(-1), weight=weights)

        return (loss, outputs) if return_outputs else loss
    