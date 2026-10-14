import os
import glob
import whisperx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
from ASR.utils import whisperX2csv

def whisperX_main(list_files_index = ['_af-1wROuhw'], # list of indexes
                  path_wav = '/home/user/data/standup/audio/test_fr/',
                  path_csv = '/home/user/data/standup/transcript/test_fr/whisperx-large-v3-french/',
                  model_name = "bofenghuang/whisper-large-v3-french", # can use "large-v3", or any model converted previsouly to fast-whisper format
                  lang=None,
                  device = "cuda",
                  batch_size = 32, # reduce if low on GPU mem
                  compute_type = "float16", # change to "int8" if low on GPU mem (may reduce accuracy)
                  on_error=None,
                  ):
    
    # Create output folder if not existing
    os.makedirs(path_csv, exist_ok=True)

    if lang == None:
        lang = path_wav.split('/')[-2].split('_')[0] # do not work for test_fr though...
        print(f'Automatically adding lang as: {lang}')

    # this was used to debug, not sure it is useful. Just converting to fast whisper seems to solve the prob
    # from huggingface_hub import snapshot_download
    # model_path = snapshot_download(model_name)
    if model_name == 'large-v3':
        model_name = runtime.local_model('Systran/faster-whisper-large-v3')
    model = whisperx.load_model(model_name, device, compute_type=compute_type,
                               language=lang, vad_method="silero")
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF
    alignment_name = DEFAULT_ALIGN_MODELS_HF.get(lang)
    if alignment_name:
        alignment_name = runtime.local_model(alignment_name)
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=device,
                                                 model_name=alignment_name)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    for fn in list_files_index:
        target = Path(path_csv) / f'{fn}.csv'
        if target.exists():
            print(f'Skip file... {target} already existing')
            continue
        try:
            audio_file = str(Path(path_wav) / f'{fn}.wav')
            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=batch_size)
            result = whisperx.align(result['segments'], model_a, metadata, audio, device, return_char_alignments=False)
            df = whisperX2csv(result)
            temporary = target.with_suffix('.tmp.csv')
            df.to_csv(temporary)
            temporary.replace(target)
        except Exception as error:
            if on_error is None:
                raise
            on_error(fn, error)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transcribe and align WAV files with WhisperX')
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--model', default='large-v3')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    whisperX_main([p.stem for p in sorted(args.input_dir.glob('*.wav'))],
                  str(args.input_dir) + os.sep, str(args.output_dir) + os.sep,
                  model_name=args.model, lang=args.language, device=args.device,
                  batch_size=args.batch_size,
                  compute_type='int8' if args.device == 'cpu' else 'float32')
