import whisper
import torch
from whisperx.vads import Silero
from pyannote.core import  Segment
from transformers import WhisperTokenizer
from difflib import SequenceMatcher
import statistics

from settings import *
from model import *
from dataset import annotation2tensor
from decoding import *

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="en", task="transcribe")
vad_options = {
        "chunk_size": 30, # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.500,
        "vad_offset": 0.363
}
vad_model = Silero(**vad_options)


def get_segments(lines):
    result = []
    for line in lines:
        if len(line) < 8:
            continue
        spk = line[-2]
        content = line[:-7]
        result.append({"speaker":spk, "text": content})
    return result


def transcribe_segment(waveform, start, end, annotation, speakers, model, device):
    start_frame = int(start * SAMPLE_RATE)
    end_frame = int(end * SAMPLE_RATE)
    audio = waveform[start_frame:end_frame]
    audio = whisper.pad_or_trim(audio, CHUNK_LENGTH * SAMPLE_RATE)
    mel = whisper.log_mel_spectrogram(audio)
    segment_annotation = annotation.crop(Segment(start, end))
    segment_annotation = annotation2tensor(segment_annotation, speakers, start)
    
    tokens = tokenizer.encode("")[:-1]
    task = DecodingWithSpeakerLabels(model, tokens, 8, 50257)
    with torch.no_grad():
        mel = mel.to(device).unsqueeze(0)
        segment_annotation = segment_annotation.to(device).unsqueeze(0)
        audio_features = model.embed_audio(mel, segment_annotation)
        tokens = task.run(audio_features)
        #candidates = beam_search_decode(model, audio_features, tokens, 50257, 8)
        #tokens, _, _ = candidates[0]
    text = tokenizer.decode(tokens)
    #print(text)
    result = get_segments(text.split("\n"))
    return result

def transcribe(waveform, annotation, model, device):
    segments = vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    segments = vad_model.merge_chunks(segments, CHUNK_LENGTH)
    speakers = {}
    result = []
    for segment in segments:
        tmp = transcribe_segment(waveform, segment["start"], segment["end"], annotation, speakers, model, device)
        result.extend(tmp)
    return result

def align(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=float)

    for j in range(1, n+1):
        dp[0][j] =  j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 1 - SequenceMatcher(None, seq1[i-1], seq2[j-1]).ratio()
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + min(1, 2 * cost)   # substitution
            )
   
    aligned = []
    i, j = len(seq1), len(seq2)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if dp[i-1][j] < min(dp[i][j-1], dp[i-1][j-1]):
                aligned.append((i-1, None))
                i -= 1
            elif dp[i][j-1] < min(dp[i-1][j], dp[i-1][j-1]):
                aligned.append((None, j-1))
                j -= 1
            else:
                aligned.append((i-1, j-1))
                i -= 1
                j -= 1
        elif i <= 0:
            aligned.append((None, j-1))
            j -= 1
        else:
            aligned.append((i-1, None))
            i -= 1
    aligned.reverse()
    return aligned

def aggregate(overlapping_result):
    candidates = []
    for segment in overlapping_result[0]["window"]:
        candidates.append({"candidates_spk":[segment["speaker"]], "text":segment["text"]})
    start_idx = 0
    cur_end = overlapping_result[0]["end"]
    
    for window in overlapping_result:

        if window["start"] >= cur_end:
            cur_end = window["end"]
            start_idx = len(candidates)
            for segment in window["window"]:
                candidates.append({"candidates_spk":[segment["speaker"]], "text":segment["text"]})
                
        else:
            pre_tokens = []
            pre_tokens_idx = []
            segment_len = []
            max_offset = len(candidates)
            for offset in range(start_idx, max_offset):
                tokens = candidates[offset]["text"].split(" ")
                pre_tokens.extend(tokens)
                segment_len.append(len(tokens))
                for _ in tokens:
                    pre_tokens_idx.append(offset)
            cur_tokens = []
            cur_tokens_idx = []
            for idx, segment in enumerate(window["window"]):
                tokens = segment["text"].split(" ")
                cur_tokens.extend(tokens)
                for _ in tokens:
                    cur_tokens_idx.append(idx)
                
            align_result = align(pre_tokens, cur_tokens)
          
            speakers_stat = [[] for _ in range(start_idx, max_offset)]
            for i, j in align_result:
                if i is None or j is None:
                    continue
                offset = pre_tokens_idx[i]
                spk = window["window"][cur_tokens_idx[j]]["speaker"]
                speakers_stat[offset-start_idx].append(spk)
            for offset in range(start_idx, max_offset):
                if speakers_stat[offset-start_idx] == []:
                    continue
                spk = statistics.mode(speakers_stat[offset-start_idx])
                if speakers_stat[offset-start_idx].count(spk) >= 0.5 * segment_len[offset-start_idx]:
                    candidates[offset]["candidates_spk"].append(spk)
            
            
            for i, j in align_result:
                if j is not None and i is not None:
                    break
            if i is not None:
                
                start_idx = min(pre_tokens_idx[i], start_idx+1)
                
                    
            pre_j = None
            
            for i, j in reversed(align_result):
                if i is not None:
                    break
                pre_j = j
            if pre_j is not None and pre_j < len(cur_tokens)-1:
                extra_start = pre_j
                cur_end = window["end"]
            else:
                extra_start = len(cur_tokens)

            new_sgement = {"text":""}
            for i in range(extra_start, len(cur_tokens)):
                if cur_tokens[i] == "":
                    continue
                new_sgement["text"] += cur_tokens[i]
                new_sgement["candidates_spk"] = [window["window"][cur_tokens_idx[i]]["speaker"]]
                
                if cur_tokens[i][-1] in [".", "?"]:
                    candidates.append(new_sgement)
                    new_sgement = {"text":""}
                else:
                    new_sgement["text"] += " "

        
    result = []
    for segment in candidates:
        spk = statistics.mode(segment["candidates_spk"])
        result.append({"speaker":spk, "text":segment["text"]})
        
    return result
            

def transcribe_overlap_voting(waveform, annotation, model, device):
    segments = vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    speakers = {}
    overlapping_result = []
    for i in range(len(segments)):
        j = i+1
        start = segments[i].start
        end = segments[i].end
        while j < len(segments) and segments[j].end - start < CHUNK_LENGTH:
            end = segments[j].end
            j += 1
        window = transcribe_segment(waveform, start, end, annotation, speakers, model, device)
        overlapping_result.append({"window":window, "start":start, "end":end})
    
    return aggregate(overlapping_result)