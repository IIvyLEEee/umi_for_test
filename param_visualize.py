import torch

# ✅ 替换成你的 checkpoint 路径
# ckpt_path = "/home/acts00/Desktop/umi/data/models/cup_wild_vit_l_1img.ckpt"
ckpt_path = "/home/acts00/Desktop/scalinglaw/data/checkpoints/pour_water/latest.ckpt"

# ✅ 加载 state_dict
state_dict = torch.load(ckpt_path, map_location='cpu')

# ✅ 如果是直接是state_dict，就用这个；如果是封装在'data'或'checkpoint'中，要改这里
# if "state_dict" in state_dict:
state_dict = state_dict['state_dicts']['model']

# ✅ 定义模块前缀
prefixes = ["obs_encoder", "model", "normalizer"]

# ✅ 初始化统计字典
stats = {prefix: {"params": 0, "bytes": 0} for prefix in prefixes}
stats["others"] = {"params": 0, "bytes": 0}  # 统计未归类参数

# ✅ 遍历每个参数
for name, tensor in state_dict.items():
    found = False
    for prefix in prefixes:
        if name.startswith(prefix):
            n_params = tensor.numel()
            stats[prefix]["params"] += n_params
            stats[prefix]["bytes"] += n_params * 4  # float32 = 4 bytes
            found = True
            break
    if not found:
        n_params = tensor.numel()
        stats["others"]["params"] += n_params
        stats["others"]["bytes"] += n_params * 4

# ✅ 打印结果（单位对齐）
print(f"{'Module':<15} {'Params':>15} {'Size (MB)':>15}")
print("-" * 45)
for module, s in stats.items():
    size_mb = s["bytes"] / (1024 ** 2)
    print(f"{module:<15} {s['params']:>15,} {size_mb:>15.2f}")
