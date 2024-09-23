import torch

# cuda 사용가능 여부
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')

# 예시로 텐서를 생성하고 장치로 이동
x = torch.randn(3, 3)
x = x.to(device)

print(x)