import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def get_free_gpu_memory():
    """获取每张 GPU 的可用显存（单位：MB）"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE
    )
    free_memory = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
    return free_memory

def run_python_script(task_id, script_path, model_path, viewpoint, rotation, gpu_id):
    """
    运行一个 Python 脚本，传递参数。
    
    :param task_id: 任务 ID
    :param script_path: 要运行的 Python 脚本路径
    :param model_path: 模型路径参数
    :param viewpoint: 视角参数
    :param rotation: 旋转参数
    :return: 任务结果
    """
    print(f"任务 {task_id} 开始，使用 GPU {gpu_id}")

    # 设置环境变量 CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    # 构造命令
    command = [
        "python", script_path,
        '--env.target_path',
        model_path,
        '--env.viewpoint_index',
        str(viewpoint),
        '--env.offset_phi_index',
        str(rotation)
    ]

    # 运行子进程
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 输出任务结果
    if result.returncode == 0:
        print(f"任务 {task_id} 完成")
        return result.stdout.strip()
    else:
        print(f"任务 {task_id} 失败: {result.stderr.strip()}")
        return None

def log_failed_task(task_id, model_path, viewpoint, rotation):
    """将失败的任务参数记录到文件"""
    with open("/home/zhengquan/04-fep-nbv/failed_tasks.log", "a") as f:
        f.write(f"任务 {task_id}: model_path={model_path}, viewpoint={viewpoint}, rotation={rotation}\n")


if __name__ == "__main__":
    # 获取每张 GPU 的可用显存
    free_memory = get_free_gpu_memory()
    free_memory = [24000]
    task_memory_usage = 6000

    script_path = '/home/zhengquan/04-fep-nbv/data/code/single_rotation_distribution.py'  # 子任务 Python 脚本路径
    viewpoints = [i for i in range(48)]                          # 总任务数
    rotations = [i for i in range(8)]
    max_workers = 6                            # 并行任务数
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a9b552befd6306cc8f2d5fe7449af61'

    # 计算每张 GPU 可并行任务数
    max_tasks_per_gpu = [free // task_memory_usage for free in free_memory]
    total_max_workers = sum(max_tasks_per_gpu)

    print(f"每张 GPU 可运行任务数: {max_tasks_per_gpu}")
    print(f"总并行任务数: {total_max_workers}")

    with ProcessPoolExecutor(max_workers=total_max_workers) as executor:
        futures = []
        gpu_index = 0
        for viewpoint in viewpoints:
            for rotation in rotations:
                task_id = rotation+viewpoint*8
                # 分配 GPU（轮询分配）
                gpu_id = gpu_index % len(max_tasks_per_gpu) + 2
                gpu_index += 1
                # 提交任务
                futures.append(
                    executor.submit(run_python_script, task_id, script_path, model_path, viewpoint, rotation, gpu_id)
                )

            # 处理任务结果
            for future in as_completed(futures):
                try:
                    output = future.result()
                    if output:
                        print(f"任务输出：\n{output}")
                except Exception as e:
                    print(f"任务执行异常: {e}")
                    log_failed_task(task_id, model_path, viewpoint, rotation)