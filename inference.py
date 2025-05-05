from whisperx.vads import Silero
import whisper
import torch
import argparse
from safetensors.torch import load_file
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from transformers import WhisperTokenizer

from settings import *
from model import *
from dataset import annotation2tensor
from transcribe import *


parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file (e.g., .mp3)")
parser.add_argument("--rttm_path", type=str, required=False, default=None, help="Path to the RTTM file (optional)")
parser.add_argument("--output_path", type=str, required=True, help="Path to the output file (e.g., .txt)")
parser.add_argument("--model_path", type=str, required=False, default="models/model.safetensors", help="Path to pretrained model")
parser.add_argument("--uri", type=str, required=False, default=None, help="Uri of rttm file")
parser.add_argument("--multi_prediction", type=bool, required=False, default=False)
parser.add_argument("--use_auth_token", type=str, required=False, default=None, help="Huggingface access code")

# Parse arguments
args = parser.parse_args()

# Access the arguments
audio_path = args.audio_path
rttm_path = args.rttm_path
output_path = args.output_path
uri = args.uri


waveform = whisper.load_audio(audio_path)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = WhisperWithSpeakers(whisper.load_model("base").dims)
model.load_state_dict(load_file(args.model_path))
model.to(device)

if rttm_path is None or uri is None:
        pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=args.use_auth_token)
        pipeline.to(device)
        annotation:Annotation = pipeline(audio_path)

else:
        annotation:Annotation = load_rttm(rttm_path)[uri]
        
if args.multi_prediction:
        result = transcribe_overlap_voting(waveform, annotation, model, device)

else:
        result = transcribe(waveform, annotation, model, device)


with open(output_path, "w") as f:
        for segment in result:
                speaker = segment["speaker"]
                text = segment["text"]
                f.write(f"Speaker{speaker}: {text}\n")
                
