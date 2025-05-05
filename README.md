# Whisper Finetuning for Speaker Diarization

In this project, we adapt whisper model to accept speaker diarization results (in rttm format), and generates transcript with speaker labels.  

To work with this project, clone the repo first:  
```
git clone https://github.com/leinace1001/whisper-finetune-speaker-diarization.git
cd whisper-finetune-speaker-diarization
```

## Inference
You can run inference with the following command:  
```
python inference.py --audio_path example.wav --output_path example.txt
```
Or, if you already have a speaker diarization rttm file, run:
```
python inference.py --audio_path example.wav --rttm_path example.rttm --uri example --output_path example.txt
```
