import json
import os
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
import whisper
from transformers import WhisperTokenizer
import random

from settings import *
import torch
from pyannote.metrics.matcher import GreedyMapper

def spk2label(speakers, label):
    if label not in speakers:
        speakers[label] = len(list(speakers.keys()))
    return speakers[label]

def spk2label_shuffle(speakers, label):
    if label not in speakers:
        used_label = []
        for key in speakers:
            used_label.append(speakers[key])
        choices = []
        for i in range(MAX_SPK_NUM):
            if i not in used_label:
                choices.append(i)
        if len(choices) == 0:
            return 0
        speakers[label] = random.choices(choices)[0]

    return speakers[label]
    
def annotation2tensor(annotation:Annotation, speakers, chunk_start, shuffle=False, mapping=None):
    result = torch.zeros(CHUNK_LENGTH*50, dtype=torch.int32)
    for spk in annotation.labels():
        gt_label = None
        if mapping:
            gt_label = mapping.get(spk)
        if gt_label is None:
            gt_label = spk
        if shuffle:
            label = spk2label_shuffle(speakers, gt_label)
        else:
            label = spk2label(speakers, gt_label)

        for segment in annotation.label_timeline(spk):
            start = int((segment.start - chunk_start) * 50)
            end = int((segment.end - chunk_start) * 50)
            if start >= CHUNK_LENGTH*50-1:
                continue
            end = min(end, CHUNK_LENGTH*50-1)
            if result[start] != 0:
                result[start+1] = 2 * label + 1
            else:
                result[start] = 2 * label + 1
            if result[end] != 0 and end < CHUNK_LENGTH*50-1:
                result[end+1] = 2 * label + 2
            else:
                result[end] = 2 * label + 2
    return result
        
def segment2annotation(segments):
    annotation = Annotation()
    for entry in segments:
        start = entry['start']
        end = entry['end']
        speaker = entry['speaker']
    
    
    # Create a Segment for the current entry
        segment = Segment(start, end)
    
    # Add the segment to the annotation with the speaker as the label
        annotation[segment] = speaker
        
    return annotation


def process_file(segment_path, rttm_path, audio_path, uri, n_mels):
    wave = whisper.load_audio(audio_path, sr=SAMPLE_RATE)
    with open(segment_path, "r") as f:
        segments = json.load(f)
    annotation:Annotation = load_rttm(rttm_path)[uri]
    gt_annotation = segment2annotation(segments)
    mapper = GreedyMapper()
    mapping = mapper(annotation, gt_annotation)
    processed = []
    
    for i, segment in enumerate(segments):
        if i % 6 != 0:
            continue
        if segment["end"] - segment["start"] >= CHUNK_LENGTH:
            continue
        speakers = {}
        
        start = segment["start"]
        end = segment["end"]
        speaker = spk2label_shuffle(speakers, segment["speaker"])
        content =  segment["text"] + f" <spk{speaker}>" + "\n"
        j = i+1
        while j < len(segments) and segments[j]["end"] - start < CHUNK_LENGTH:
            end = segments[j]["end"]
            speaker = spk2label_shuffle(speakers, segments[j]["speaker"])
            content += segments[j]["text"] + f" <spk{speaker}>"+"\n"
            j += 1
            
        if len(content) > 0:
            audio = wave[int(start*SAMPLE_RATE):int(end*SAMPLE_RATE)]
            audio = whisper.pad_or_trim(audio, CHUNK_LENGTH*SAMPLE_RATE)
            mel = whisper.log_mel_spectrogram(audio, n_mels)
            segment_annotation = annotation.crop(Segment(start, end))
            sd_hypothesis = annotation2tensor(segment_annotation, speakers, start, mapping)
            processed.append({"content":content, "mel":mel, "speakers":sd_hypothesis})
            
    
    return processed


def ami_data_finder(words_root, audio_root, rttm_root, uris):
    output = []
    for uri in uris:
        uri = uri.strip()
        words_path = os.path.join(words_root, uri+".json")
        rttm_path = os.path.join(rttm_root, uri+".rttm")
        audio_path = os.path.join(audio_root, uri, "audio", f"{uri}.Mix-Headset.wav")
        if os.path.exists(words_path) and os.path.exists(rttm_path) and os.path.exists(audio_path):
            output.append((words_path, rttm_path, audio_path, uri))
    
    return output

def custom_data_finder_with_uri(words_root, audio_root, rttm_root, uris):
    output = []
    for uri in uris:
        uri = uri.strip()
        words_path = os.path.join(words_root, uri+".json")
        rttm_path = os.path.join(rttm_root, uri+".rttm")
        audio_path = os.path.join(audio_root, f"{uri}.wav")
        if os.path.exists(words_path) and os.path.exists(rttm_path) and os.path.exists(audio_path):
            output.append((words_path, rttm_path, audio_path, uri))
    return output

def custom_data_finder(words_root, audio_root, rttm_root, uris):
    # modify this function if needed
    output = []
    for root, _, files in os.walk(words_root):
        for file in files:
            if file[-4:] != "json":
                continue
            file_path = os.path.join(root, file)
            audio_path = file_path.replace(words_root, audio_root)
            audio_path = audio_path[:-4] + "mp3"
            rttm_path = file_path.replace(words_root, rttm_root)
            rttm_path = rttm_path[:-4] + "rttm"
            
            if os.path.exists(rttm_path) and os.path.exists(audio_path):
                output.append((file_path, rttm_path, audio_path, "waveform"))
    return output

def all_data(words_root, audio_root, rttm_root, n_mels, uri_file=None):
    uris = None
    if uri_file is not None:
        with open(uri_file, "r") as f:
            uris = f.readlines()
    all_paths = custom_data_finder(words_root, audio_root, rttm_root, uris) # call other data finder function if needed
    result = {"content":[], "mel":[], "speakers":[]}
    
    for (words_path, rttm_path, audio_path, uri) in all_paths:
        tmp = process_file(words_path, rttm_path, audio_path, uri, n_mels)
        for item in tmp:
            result["content"].append(item["content"])
            result["mel"].append(item["mel"])
            result["speakers"].append(item["speakers"])
    return result
    

def process_batch(examples):
    """
    A batch processor that:
      - Tokenizes text
      - Keeps audio features as-is
      - Converts speaker labels into a field for the model
    """
    # 'examples' is a dict of lists (one list per column) because 'batched=True' is used.

    # 2. Tokenize the text in the batch (list of strings)
    #    return_tensors="pt" is optional here; we can return plain Python lists
    #    to keep it compatible with dataset mapping.
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="en", task="transcribe")
    text_encoding = tokenizer(examples["content"], 
                            padding="max_length",
                            max_length = DECODE_MAX_LENGTH,
                            truncation=True,
                            return_tensors="pt")
    

    # 3. Prepare model inputs
    #    Here we create a new dictionary that merges tokenized text
    #    with audio features and speaker labels.
    #print(text_encoding["input_ids"].shape)
    batch = {}

    batch["tokens"] = []
    for i in range(len(examples["content"])):
        batch["tokens"].append(text_encoding["input_ids"][i])
    # 4. Keep audio features as-is
    batch["mel"] = examples["mel"]
    # 5. Store speaker labels (could be int or string).
    #    Rename it to "speaker_labels" or "labels", depending on your training setup.
    batch["speakers"] = examples["speakers"]

    return batch
            



            
