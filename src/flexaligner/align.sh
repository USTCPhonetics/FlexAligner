#!/bin/bash

python ./scripts/chunks2.py \
  --device "cpu" \
  --model_dir ./models/hf_phs \
  --lexicon ./dictionary/dict.mandarin.2 \
  --phone_json ./models/hf_phs/vocab.json \
  --blank_token "<pad>" \
  --audio ./test/SP01_001.wav \
  --transcript ./test/SP01_001.txt \
  --out_dir ./chunks_out \
  --max_gap_s 0.35 \
  --min_chunk_s 1.0 \
  --max_chunk_s 12.0 \
  --pad_s 0.15 \
  --chunk_audio

python ./scripts/align_chunks_and_combine_textgrids.py \
  --chunks_tsv ./chunks_out/chunks/SP01_001.chunks.tsv \
  --transcript_col words \
  --chunk_wav_dir ./chunks_out/chunks \
  --chunk_wav_glob "{chunk_id}_*.wav" \
  --time_match_tol_s 0.01 \
  --full_wav  ./test/SP01_001.wav \
  --work_dir ./work_align \
  --align_script ./scripts/align_ce_w2v2.py \
  --ckpt ./models/ce2 \
  --lexicon ./dictionary/dict.mandarin.2 \
  --optional_sil --sil_at_ends \
  --beam 400 --p_stay 0.92 --frame_hop_s 0.01 \
  --nj 8 \
  --out ./test/SP01_001.TextGrid \
  --tsv_out_prefix ./test/SP01_001 \
  --tsv_tiers words phones \

rm -r -f ./chunks_out
rm -r -f ./work_align
