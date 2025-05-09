
import os
import whisperx
import json
import numpy as np
import string
import torchaudio
from whisperx.vads import Silero

SAMPLE_RATE = 16000
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)
audio_root = "/local/scratch/mli526/dataset/DW/audio"

output_folder = "/local/scratch/mli526/dataset/DW/whisper"
device = "cuda"

batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

asr_options = {
    "initial_prompt": "Hola, ¿cómo estás? We're switching between English y español en esta conversación. Sometimes hablamos en inglés, and other times cambiamos al español. ",
    #"condition_on_previous_text": True,
}
vad_options = {
        "chunk_size": 15, # needed by silero since binarization happens before merge_chunks
        "vad_onset": 0.5,
        "vad_offset": 0.36
}
model = whisperx.load_model("large-v3", device, compute_type=compute_type, vad_method="silero", asr_options=asr_options, vad_options=vad_options)
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
#vad = Silero(**vad_options)
os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(audio_root):
    output_root = root.replace(audio_root, output_folder)
    for dir in dirs:
        output_dir = os.path.join(output_root, dir)
        os.makedirs(output_dir, exist_ok=True)
    for file in files:
        if file[-3:] != 'mp3' and file[-3:] != 'wav':
            continue
        #if file[:-4] not in ["DW_A0013", "DW_A0024"]:
        #    continue
        output_path = os.path.join(output_root, file[:-3]+"txt")
        audio_path = os.path.join(root, file)
        print(audio_path)
        #ori_audio, sr = torchaudio.load(audio_path)
        # channel1 = ori_audio[0,:].unsqueeze(0)
        # channel1_path = os.path.join(cache_dir, file[:-4]+"_1.mp3")
        # torchaudio.save(channel1_path, channel1, sr)
        audio = whisperx.load_audio(audio_path)
        #print(audio.shape)
        all_results = []
        
        # segments = vad({"waveform": audio, "sample_rate": SAMPLE_RATE})
        # curr_start = 0
        # curr_end = 0
        # curr_lang = None
 
        # for segment in segments:
        #     start_frame = int(segment.start * SAMPLE_RATE)
        #     end_frame = int(segment.end * SAMPLE_RATE)
        #     lan = model.detect_language(audio[start_frame:end_frame])
        #     lan = lan if lan in ["es", "en"] and (segment.end-segment.start)>0.5 else curr_lang
        #     if lan == curr_lang or curr_lang is None:
        #         curr_end = segment.end
        #     else:
        #         start_frame = int(curr_start * SAMPLE_RATE)
        #         end_frame = int(curr_end * SAMPLE_RATE)
        #         print(curr_end-curr_start)
        #         results = model.transcribe(audio[start_frame:end_frame], batch_size=batch_size, language=curr_lang)
        #         for result in results["segments"]:
        #             result["start"] += curr_start
        #             result["end"] += curr_start
        #             all_results.append(result)
        #         curr_start = segment.start
        #         curr_end = segment.end
        #     curr_lang = lan
            
        # start_frame = int(curr_start * SAMPLE_RATE)
        # end_frame = int(curr_end * SAMPLE_RATE)
        # results = model.transcribe(audio[start_frame:end_frame], batch_size=batch_size)
        # for result in results["segments"]:
        #     result["start"] += curr_start
        #     result["end"] += curr_start
        #     all_results.append(result)
      
        #span_segments = vad.merge_chunks(spanish_chunks, 15)
        #eng_segments = vad.merge_chunks(english_chunks, 15)
        # for segment in span_segments:
        #     start_frame = int(segment["start"] * SAMPLE_RATE)
        #     end_frame = int(segment["end"] * SAMPLE_RATE)
        #     results = model.transcribe(audio[start_frame:end_frame], batch_size=batch_size)
        #     all_results.extend(results["segments"])
            
        # for segment in eng_segments:
        #     start_frame = int(segment["start"] * SAMPLE_RATE)
        #     end_frame = int(segment["end"] * SAMPLE_RATE)
        #     results = model.transcribe(audio[start_frame:end_frame], batch_size=batch_size)
        #     all_results.extend(results["segments"])
        results = model.transcribe(audio, batch_size=batch_size)

        results = whisperx.align(results["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        with open(output_path, "w") as f:
            for segment in results["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                f.write(f"{start:.2f}\t{end:.2f}\t{text}\n")
        #with open(output_json_path, "w") as f:
            #json.dump(results, f)
    

        # for segment in output_segments:
        #     segment["speaker"] = "1"
        # channel2 = ori_audio[1,:].unsqueeze(0)
        # channel2_path = os.path.join(cache_dir, file[:-4]+"_2.mp3")
        # torchaudio.save(channel2_path, channel2, sr)
        # audio = whisperx.load_audio(channel2_path)
        # results = model.transcribe(audio, batch_size=batch_size)
        # output_segments2 = whisperx.align(results["segments"], model_a, metadata, audio, device, return_char_alignments=False)["segments"]
        # for segment in output_segments2:
        #     segment["speaker"] = "2"
        # output_segments.extend(output_segments2)
        # output_segments.sort(key=lambda x:x["start"])
    


        

