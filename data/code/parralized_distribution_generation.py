import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import time
from multiprocessing import Manager, Lock
root_path = os.getenv('nbv_root_path', '/default/path')
shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')


def get_free_gpu_memory():
    """获取每张 GPU 的可用显存（单位：MB）"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE
    )
    free_memory = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
    return free_memory

def run_python_script(task_id, script_path, model_path, viewpoint, rotation, gpu_id, running_tasks, task_lock):
    """
    运行一个 Python 脚本，传递参数。
    
    :param task_id: 任务 ID
    :param script_path: 要运行的 Python 脚本路径
    :param model_path: 模型路径参数
    :param viewpoint: 视角参数
    :param rotation: 旋转参数
    :return: 任务结果
    """
    with task_lock:
        running_tasks.value += 1
        print(f"任务 {task_id} 开始，使用 GPU {gpu_id}，当前运行任务数: {running_tasks.value}")


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
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    # 使用 Popen 进行非阻塞子进程调用
    # result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # stdout, stderr = result.communicate()

    with task_lock:
        running_tasks.value -= 1

    if result.returncode == 0:
        print(f"任务 {task_id} 完成，当前运行任务数: {running_tasks.value}")
        return result.stdout.strip()
    else:
        print(f"任务 {task_id} 失败: {result.stderr.strip()}，当前运行任务数: {running_tasks.value}")
        return None

def log_failed_task(task_id, model_path, viewpoint, rotation):
    """将失败的任务参数记录到文件"""
    with open(root_path+"/nohup_log/failed_tasks.log", "a") as f:
        f.write(f"任务 {task_id}: model_path={model_path}, viewpoint={viewpoint}, rotation={rotation}\n")


if __name__ == "__main__":
    time1 = time.time()

    manager = Manager()
    running_tasks = manager.Value('i', 0)
    task_lock = manager.Lock()

    # 获取每张 GPU 的可用显存
    free_memory = get_free_gpu_memory()
    # free_memory = [24000]
    task_memory_usage = 4800

    script_path = os.path.join(root_path,'data/code/single_rotation_distribution.py')  # 子任务 Python 脚本路径
    viewpoints = [i for i in range(48)]                          # 总任务数
    rotations = [i for i in range(8)]
    max_workers = 6                            # 并行任务数
    model_path = os.path.join(shapenet_path,'02691156/1a32f10b20170883663e90eaf6b4ca52')

    # 计算每张 GPU 可并行任务数
    max_tasks_per_gpu = [free // task_memory_usage for free in free_memory]
    total_max_workers = sum(max_tasks_per_gpu)
    # total_max_workers = 5

    print(f"每张 GPU 可运行任务数: {max_tasks_per_gpu}")
    print(f"总并行任务数: {total_max_workers}")

    gpu_task_count = manager.dict({i: 0 for i in range(len(get_free_gpu_memory()))})

    with ProcessPoolExecutor(max_workers=total_max_workers) as executor:
        futures = []
        gpu_index = 0
        for viewpoint in viewpoints:
            for rotation in rotations:
                task_id = rotation+viewpoint*8
                # **循环等待有可用的 GPU**
                while True:
                    with task_lock:
                        if running_tasks.value < total_max_workers:
                            break
                    print("任务过多，等待 1 分钟...")
                    time.sleep(60)
                while True:
                    free_memory = get_free_gpu_memory()
                    available_gpus = [
                        i for i, mem in enumerate(free_memory)
                        if mem > task_memory_usage and gpu_task_count[i] < max_tasks_per_gpu[i]
                    ]

                    if available_gpus:
                        break  # 找到符合条件的 GPU，退出循环
                    else:
                        print("所有 GPU 都没有可用资源，等待 1 分钟后重试...")
                        time.sleep(60)  # 等待 5 分钟后重试
                # **随机选择一个符合条件的 GPU**
                gpu_id = random.choice(available_gpus)
                gpu_task_count[gpu_id] += 1  # 增加任务计数

                # 提交任务
                future = executor.submit(run_python_script, task_id, script_path, model_path, viewpoint, rotation, gpu_id, running_tasks, task_lock)

                def task_done_callback(fut, gpu_id=gpu_id):
                    gpu_task_count[gpu_id] -= 1

                future.add_done_callback(task_done_callback)
                futures.append(future)


            # 处理任务结果
            for future in as_completed(futures):
                try:
                    output = future.result()
                    # if output:
                    #     print(f"任务输出：\n{output}")
                except Exception as e:
                    print(f"任务执行异常: {e}")
                    log_failed_task(task_id, model_path, viewpoint, rotation)

    time2 = time.time()
    print(f"任务全部完成，总耗时: {((time2 - time1)/60/60):.2f} 小时")