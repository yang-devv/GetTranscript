import os
import glob
import sys
import time
import re
from datetime import datetime
from faster_whisper import WhisperModel, BatchedInferencePipeline


# ──────────────────────────── Logging ────────────────────────────

LOG_FILE = "transcribe_wav_log.txt"

def log(msg: str, level: str = "INFO"):
    """Print timestamped message to console and append to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] [{level}] {msg}"
    try:
        encoding = sys.stdout.encoding or "utf-8"
        print(full_msg.encode(encoding, errors="replace").decode(encoding))
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
    except Exception:
        pass


def format_duration(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ──────────────────────────── Main ───────────────────────────────

def main():
    model_size = "large-v3"

    # ── GPU / CPU detection ──
    import ctranslate2
    cuda_device_count = ctranslate2.get_cuda_device_count()
    if cuda_device_count > 0:
        device = "cuda"
        compute_type = "int8_float16"   # fastest on RTX 4090
        log(f"GPU detected! Found {cuda_device_count} CUDA device(s).")
        log(f"Loading {model_size} model on GPU (device={device}, compute={compute_type})...")
    else:
        device = "cpu"
        compute_type = "int8"
        log("WARNING: No GPU detected! Falling back to CPU — this will be very slow.", "WARN")
        log(f"Loading {model_size} model on CPU...")

    # ── Load model + batched pipeline ──
    # num_workers > 1 lets the data-loader pipeline keep the GPU fed continuously
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=4,          # parallel CPU feature-extraction workers
    )

    # BatchedInferencePipeline dramatically increases GPU throughput by
    # batching multiple audio chunks together in one CUDA call.
    batched_model = BatchedInferencePipeline(model=model)

    input_dir  = "rawwav"
    output_dir = "transcripts_wav"
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    total = len(wav_files)
    log(f"Found {total} .wav file(s) in '{input_dir}'.")

    # ── Per-file processing ──
    skipped   = 0
    succeeded = 0
    failed    = 0
    session_start = time.perf_counter()

    for i, audio_file in enumerate(wav_files, 1):
        filename  = os.path.basename(audio_file)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")

        if os.path.exists(output_file):
            log(f"[{i}/{total}] Skipping {filename} (already transcribed)")
            skipped += 1
            continue

        log(f"[{i}/{total}] Transcribing {filename} ...")
        t_start = time.perf_counter()

        try:
            # batch_size controls how many audio chunks are processed together
            # on the GPU per forward pass — higher = more GPU utilisation
            segments, info = batched_model.transcribe(
                audio_file,
                language="zh",        # "zh" = Mandarin/Simplified Chinese (only option in Whisper)
                beam_size=5,
                batch_size=16,        # key parameter for GPU batching
                condition_on_previous_text=True,
                vad_filter=True,
                compression_ratio_threshold=2.4,
                # NOTE: initial_prompt is ignored by BatchedInferencePipeline.
                # Use hotwords instead — they ARE honoured in batched mode and
                # nudge the model toward emitting Chinese punctuation marks.
                hotwords="，。！？、；：""''（）",
            )

            # ── Collect & format segments into readable paragraphs ──
            # Chinese sentence-ending punctuation that triggers a new paragraph
            PARA_END = re.compile(r'[。！？…]+$')
            # Minimum characters before we're willing to break into a new paragraph
            MIN_PARA_CHARS = 60

            paragraphs: list[str] = []
            current: list[str] = []
            current_len = 0

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                current.append(text)
                current_len += len(text)
                # Break into a new paragraph when we hit a sentence boundary
                # AND have accumulated enough characters
                if current_len >= MIN_PARA_CHARS and PARA_END.search(text):
                    paragraphs.append("".join(current))
                    current = []
                    current_len = 0

            # Flush any remaining text
            if current:
                paragraphs.append("".join(current))

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(paragraphs) + "\n")

            elapsed = time.perf_counter() - t_start
            audio_duration = info.duration if hasattr(info, "duration") else 0
            rtf = elapsed / audio_duration if audio_duration > 0 else 0

            log(
                f"[{i}/{total}] Done {filename} | "
                f"took {format_duration(elapsed)} | "
                f"audio {format_duration(audio_duration)} | "
                f"RTF {rtf:.2f}x"
            )
            succeeded += 1

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            log(f"[{i}/{total}] ERROR transcribing {filename} after {format_duration(elapsed)}: {e}", "ERROR")
            failed += 1

    # ── Session summary ──
    total_elapsed = time.perf_counter() - session_start
    log("─" * 60)
    log(
        f"Session complete | total {format_duration(total_elapsed)} | "
        f"succeeded {succeeded} | skipped {skipped} | failed {failed}"
    )


if __name__ == "__main__":
    # Initialise log file (overwrite previous run's log)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Transcription session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    main()
