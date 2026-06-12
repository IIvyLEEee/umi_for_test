import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_util import ArraySpec
# 假设 SharedMemoryQueue 类保存在 shared_memory_queue.py 中
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty

def main():
    # 使用 SharedMemoryManager 创建共享内存上下文
    with SharedMemoryManager() as shm_manager:
        # 构造一个示例指令，这里包含一个 numpy 数组和一个数字
        example_command = {
            'cmd': np.array([1, 2, 3]),  # 指令标识或数据
            'data': 42                 # 其他参数
        }

        # 通过示例数据构造队列（内部会自动根据 example_command 中的数据类型和形状生成对应的 ArraySpec）
        buffer_size = 10  # 设置队列的缓冲区大小
        queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_command,
            buffer_size=buffer_size
        )
        print("共享内存队列创建成功。")

        # 放入一条指令到队列中
        print("向队列中放入一条指令...")
        queue.put(example_command)
        print("指令放入成功。")

        commands = queue.get_all()
        print(commands)
        # 从队列中读出一条指令
        print("从队列中读取指令...")
        retrieved_command = queue.get()
        print("读取到的指令数据：")
        for key, value in retrieved_command.items():
            print(f"{key}: {value}")

if __name__ == '__main__':
    main()
