import sys
import os
import math
import torch
import multiprocessing as mp
from pathlib import Path

# =====================================================================
# 1. 核心定址与环境配置
# =====================================================================
PROJECT_ROOT = Path("/home/wangyiming/projects/FlexAligner")
sys.path.append(str(PROJECT_ROOT / "src"))

# 强制本地模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from flexaligner.pipeline import FlexAligner
from flexaligner.config import AlignmentConfig

# =====================================================================
# 2. 并行工作单元 (Worker)
# =====================================================================
def align_worker(worker_id, gpu_id, tasks, config_params):
    """
    每个进程独立的对齐单元
    """
    # 物理隔离显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 在子进程内重新加载必要的环境变量
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # 构造该进程专用的配置
    config = AlignmentConfig(
        lang=config_params['lang'],
        # 隔离后，该进程看到的显卡永远是 cuda:0
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        
        chunk_model_path=config_params['chunk_model_path'],
        align_model_path=config_params['align_model_path'],
        lexicon_path=config_params['lexicon_path'],
        
        # 物理及算法参数
        max_gap_s=0.35,
        min_chunk_s=1.0,
        max_chunk_s=12.0,
        pad_s=0.15,
        align_beam_size=400,
        p_stay=0.92,
        frame_hop_s=0.01,
        
        # 🔴 [关键] 开启 G2P 以解决 pizzerias 等 OOV 报错
        # 这里的 config 会被传入你提供的 TextFrontend
        # use_g2p=True 
    )

    # 实例化引擎 (每个进程一套，显存占用约 2GB)
    try:
        aligner = FlexAligner(config)
        # 屏蔽子进程的 verbose 打印，防止终端日志炸掉
        aligner.config_dict["verbose"] = False 
        
        # 执行任务块
        aligner.align_batch(tasks)
        print(f"✅ [Worker {worker_id:02d} | GPU {gpu_id}] Finished {len(tasks)} tasks.")
    except Exception as e:
        print(f"❌ [Worker {worker_id:02d} | GPU {gpu_id}] Failed: {e}")

# =====================================================================
# 3. 主调度程序
# =====================================================================
def main():
    # 必须使用 spawn 模式以安全初始化 CUDA
    mp.set_start_method('spawn', force=True)

    # 路径配置
    AUDIO_DIR = Path("/mnt/hd/data_wangyiming/timit/timit_16000")
    ANNO_DIR = Path("/mnt/hd/data_wangyiming/timit/timit_16000")
    OUTPUT_DIR = Path("/mnt/hd/data_wangyiming/timit/flexaligner-submit_initial_dictionary")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 模型资产
    SEG_MODEL = str(PROJECT_ROOT / "models_hidden/en/hf_phs")
    ALIGN_MODEL = str(PROJECT_ROOT / "models_hidden/en/ce17.6000")
    DICT_PATH = str(PROJECT_ROOT / "assets/dictionaries/en.dict")

    # 1. 扫描任务
    all_wavs = sorted(list(AUDIO_DIR.glob("*.wav")))
    tasks = []
    for wav_path in all_wavs:
        txt_path = ANNO_DIR / f"{wav_path.stem}.txt"
        out_path = OUTPUT_DIR / f"{wav_path.stem}.TextGrid"
        if txt_path.exists():
            tasks.append((str(wav_path), str(txt_path), str(out_path)))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("❌ No tasks found!")
        return

    # 2. 兵力部署：6 张显卡，每张卡 5 个进程
    NUM_GPUS = 6
    PER_GPU_WORKERS = 5
    TOTAL_WORKERS = NUM_GPUS * PER_GPU_WORKERS

    # 切分任务块
    chunk_size = math.ceil(total_tasks / TOTAL_WORKERS)
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, total_tasks, chunk_size)]

    print(f"🚀 Igniting Parallel Engine")
    print(f"   Total Tasks: {total_tasks} | Workers: {TOTAL_WORKERS} | Chunk Size: {chunk_size}")
    print("-" * 60)

    # 配置透传
    config_params = {
        'lang': 'en',
        'chunk_model_path': SEG_MODEL,
        'align_model_path': ALIGN_MODEL,
        'lexicon_path': DICT_PATH
    }

    # 3. 启动进程池
    processes = []
    for i in range(len(task_chunks)):
        gpu_id = i % NUM_GPUS # 轮询分配：0,1,2,3,4,5,0,1...
        p = mp.Process(
            target=align_worker, 
            args=(i, gpu_id, task_chunks[i], config_params)
        )
        p.start()
        processes.append(p)

    # 4. 等待收工
    for p in processes:
        p.join()

    print("-" * 60)
    print(f"🏁 Mission Accomplished. Results at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()