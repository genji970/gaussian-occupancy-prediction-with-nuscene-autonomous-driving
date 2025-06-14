import torch

def chunked_cdist(xyz: torch.Tensor, anchor_grid: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """
    메모리 절약을 위한 chunked torch.cdist
    Args:
        xyz: (Nv, 3)
        anchor_grid: (N, 3)
        chunk_size: xyz를 나눌 단위

    Returns:
        dist: (Nv, N)
    """
    Nv = xyz.shape[0]
    N = anchor_grid.shape[0]
    dist_chunks = []

    for i in range(0, Nv, chunk_size):
        chunk = xyz[i:i + chunk_size]  # (chunk_size, 3)
        dist_chunk = torch.cdist(chunk, anchor_grid)  # (chunk_size, N)
        dist_chunks.append(dist_chunk)

    return torch.cat(dist_chunks, dim=0)  # (Nv, N)

def chunked_softmax(x: torch.Tensor, dim: int = -1, chunk_size: int = 4096):
    """
    Softmax 쪼개서 계산 (메모리 절약용)

    Args:
        x: input tensor (..., N)
        dim: softmax 적용할 차원
        chunk_size: 나눌 단위 (4096~8192 권장)

    Returns:
        Softmax 결과 tensor
    """
    # 1. exp 계산 (메모리 문제 ↓)
    chunks = torch.chunk(x, chunks=(x.shape[dim] + chunk_size - 1) // chunk_size, dim=dim)
    exp_chunks = [torch.exp(chunk) for chunk in chunks]

    # 2. 전체 exp sum 계산 (정규화용)
    exp_sum = torch.sum(torch.cat(exp_chunks, dim=dim), dim=dim, keepdim=True)

    # 3. 정규화
    softmax_chunks = [exp / exp_sum for exp in exp_chunks]

    # 4. 다시 합치기
    return torch.cat(softmax_chunks, dim=dim)

