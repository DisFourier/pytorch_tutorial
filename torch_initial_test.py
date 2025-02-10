import torch

# 检查 PyTorch 是否安装成功
print("PyTorch Version:", torch.__version__)

# 检查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 获取 GPU 设备数量
print("GPU Count:", torch.cuda.device_count())

# 获取当前 GPU 名称
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))


# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 获取当前使用的 GPU 设备
    device = torch.device("cuda")
    # 创建一个张量并将其移动到 GPU 上
    x = torch.tensor([1.0, 2.0]).to(device)
    y = torch.tensor([3.0, 4.0]).to(device)
    # 在 GPU 上进行计算
    z = x + y
    print("计算结果:", z)
    print("计算结果所在设备:", z.device)
else:
    print("CUDA 不可用")