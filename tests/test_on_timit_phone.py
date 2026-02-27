import sys
import os
import math
from pathlib import Path
import torch
import multiprocessing as mp

# [极客防线] 离线模式锁定 (在主进程先设一次)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

PROJECT_ROOT = Path("/home/wangyiming/projects/FlexAligner")
sys.path.append(str(PROJECT_ROOT / "src"))

# =====================================================================
# 🛡️ 核心手术：动态劫持前端处理
# =====================================================================
from flexaligner.frontend import TextFrontend

def raw_phoneme_bypass(self, text: str, lang: str):
    tokens = text.strip().split()
    return [w.lower() if w.upper() == "SIL" else w for w in tokens]

TextFrontend.get_phonemes = raw_phoneme_bypass
# =====================================================================

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig

def worker_process(worker_id, target_gpu, task_chunk):
    """
    独立作战单元：被分配到指定显卡，拥有自己独立的 FlexAligner 实例
    """
    # 1. 物理隔离：向该进程隐瞒其他 GPU 的存在
    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # 确保每个进程有独立的输出和日志隔离
    # print(f"[Worker {worker_id:02d} | GPU {target_gpu}] Igniting with {len(task_chunk)} tasks...")

    # 2. 核心引擎配置
    SEG_MODEL_DIR = PROJECT_ROOT / "models_hidden/en/hf_phs"
    ALIGN_MODEL_DIR = PROJECT_ROOT / "models_hidden/en/ce17.6000" 
    DICT_PATH = PROJECT_ROOT / "assets/dictionaries/dict2"
    
    config = AlignmentConfig(
        lang="en",
        # 因为已经被 CUDA_VISIBLE_DEVICES 隔离，这里直接用 cuda:0 即可映射到真实的 target_gpu
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        
        chunk_model_path=str(SEG_MODEL_DIR),
        align_model_path=str(ALIGN_MODEL_DIR),
        lexicon_path=str(DICT_PATH),
        
        min_chunk_s=1.0, max_chunk_s=12.0, max_gap_s=0.35, pad_s=0.15, blank_token="<pad>",
        sil_phone="sil", optional_sil=True, sil_cost=-0.5, align_beam_size=400, p_stay=0.92, frame_hop_s=0.01,
        boundary_lambda=200.0, boundary_context_s=0.05,
    )
    
    config.sil_at_ends = True
    config.min_sil_dur_ms = 50.0
    config.sil_enter_cost = -0.5

    # 3. 实例化独立引擎
    try:
        # 为了防止 30 个进程同时疯狂打印初始化日志导致终端崩溃，我们可在此静默处理
        aligner = FlexAligner(config)
        aligner.config_dict["sil_at_ends"] = config.sil_at_ends
        aligner.config_dict["min_sil_dur_ms"] = config.min_sil_dur_ms
        aligner.config_dict["sil_enter_cost"] = config.sil_enter_cost
        
        # 暂时关闭子进程内部的详细输出，只看进度
        aligner.config_dict["verbose"] = False 
    except Exception as e:
        print(f"❌ [Worker {worker_id:02d}] 引擎初始化失败: {e}")
        return

    # 4. 执行对齐
    try:
        aligner.align_batch(task_chunk)
        print(f"✅ [Worker {worker_id:02d} | GPU {target_gpu}] Finished {len(task_chunk)} tasks.")
    except Exception as e:
        print(f"❌ [Worker {worker_id:02d} | GPU {target_gpu}] Crashed: {e}")


def main():
    # 强制 PyTorch 使用 spawn 模式，避免多进程下的 CUDA 显存死锁
    mp.set_start_method('spawn', force=True)

    # ================= 1. 战区拓扑定址 =================
    AUDIO_DIR = Path("/mnt/hd/data_wangyiming/timit/timit_16000")
    ANNO_DIR = Path("/mnt/hd/data_wangyiming/timit/annotations")
    OUTPUT_DIR = Path("/mnt/hd/data_wangyiming/timit/flexaligner-submit-50ms-enter0_initial_dictionary_cost-0.5")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ================= 2. 收集全局任务 =================
    all_tasks = []
    wav_files = sorted(list(AUDIO_DIR.glob("*.wav")))
    
    for wav_path in wav_files:
        txt_path = ANNO_DIR / f"{wav_path.stem}.txt"
        out_path = OUTPUT_DIR / f"{wav_path.stem}.TextGrid"
        if txt_path.exists():
            all_tasks.append((str(wav_path), str(txt_path), str(out_path)))

    total_tasks = len(all_tasks)
    if total_tasks == 0:
        print("❌ No tasks found!")
        return

    # ================= 3. 计算兵力部署 (负载均衡) =================
    NUM_GPUS = 6
    PROCESSES_PER_GPU = 5
    TOTAL_WORKERS = NUM_GPUS * PROCESSES_PER_GPU

    # 将所有任务切分成 TOTAL_WORKERS 个块
    chunk_size = math.ceil(total_tasks / TOTAL_WORKERS)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_tasks, chunk_size)]

    print(f"🚀 Mass Parallel Test Ignited")
    print(f"   Total Tasks: {total_tasks}")
    print(f"   Hardware:    {NUM_GPUS} GPUs")
    print(f"   Concurrency: {TOTAL_WORKERS} Workers ({PROCESSES_PER_GPU} per GPU)")
    print("-" * 60)

    # ================= 4. 派发进程 =================
    processes = []
    for worker_id, chunk in enumerate(chunks):
        # 轮询分配 GPU: 0, 1, 2, 3, 4, 5, 0, 1...
        target_gpu = worker_id % NUM_GPUS 
        
        p = mp.Process(target=worker_process, args=(worker_id, target_gpu, chunk))
        p.start()
        processes.append(p)

    # 等待所有子进程结束
    for p in processes:
        p.join()

    print("-" * 60)
    print(f"🏁 All 30 Strike Teams have returned. Deployment Complete.")
    print(f"   Results waiting at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()