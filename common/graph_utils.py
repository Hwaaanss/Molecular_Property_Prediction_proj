from __future__ import annotations

import torch
import torch.nn as nn


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int | None = None,
) -> torch.Tensor:
    if src.dim() == 1:
        src = src.unsqueeze(-1)
    if index.numel() == 0:
        dim_size = 0 if dim_size is None else dim_size
        return src.new_zeros((dim_size, src.size(-1)))
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = src.new_zeros((dim_size, src.size(-1)))
    out.index_add_(0, index, src)
    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int | None = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    if index.numel() == 0:
        return out
    count = out.new_zeros((out.size(0), 1))
    ones = out.new_ones((index.size(0), 1))
    count.index_add_(0, index, ones)
    return out / count.clamp(min=1)


def size_norm(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if batch.numel() == 0 or x.size(0) == 0:
        return x
    counts = scatter_sum(
        torch.ones((batch.size(0), 1), device=x.device, dtype=x.dtype),
        batch,
    )
    return x / counts[batch].sqrt().clamp(min=1.0)


def build_reverse_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    edge_pairs = edge_index.t().tolist()
    edge_map = {(int(src), int(dst)): idx for idx, (src, dst) in enumerate(edge_pairs)}
    reverse_indices = [edge_map.get((int(dst), int(src)), -1) for src, dst in edge_pairs]
    return torch.tensor(reverse_indices, device=edge_index.device, dtype=torch.long)


def build_line_graph(edge_index: torch.Tensor) -> torch.Tensor:
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return edge_index.new_empty((2, 0))

    src, dst = edge_index
    reverse_index = build_reverse_edge_index(edge_index)
    line_edges: list[list[int]] = []
    for edge_idx in range(num_edges):
        incoming_edges = torch.where(dst == src[edge_idx])[0]
        if incoming_edges.numel() == 0:
            continue
        for incoming_idx in incoming_edges.tolist():
            if incoming_idx != int(reverse_index[edge_idx]):
                line_edges.append([incoming_idx, edge_idx])

    if not line_edges:
        return edge_index.new_empty((2, 0))
    return torch.tensor(
        line_edges,
        dtype=torch.long,
        device=edge_index.device,
    ).t().contiguous()


def to_dense_batch_manual(
    x: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.size(0) == 0:
        return (
            x.new_zeros((0, 0, x.size(-1))),
            x.new_zeros((0, 0), dtype=torch.bool),
        )

    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs)
    max_nodes = int(counts.max().item())
    dense = x.new_zeros((num_graphs, max_nodes, x.size(-1)))
    mask = torch.zeros((num_graphs, max_nodes), device=x.device, dtype=torch.bool)
    ptr = torch.zeros(num_graphs, dtype=torch.long, device=x.device)

    for idx in range(x.size(0)):
        graph_idx = int(batch[idx].item())
        node_pos = int(ptr[graph_idx].item())
        dense[graph_idx, node_pos] = x[idx]
        mask[graph_idx, node_pos] = True
        ptr[graph_idx] += 1

    return dense, mask


def build_dense_adjacency(
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    max_nodes: int,
) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    device = edge_index.device if edge_index.numel() > 0 else batch.device
    adjacency = torch.zeros((num_graphs, max_nodes, max_nodes), device=device)
    if edge_index.size(1) == 0:
        return adjacency

    node_pos = torch.zeros(batch.size(0), dtype=torch.long, device=batch.device)
    graph_counts = torch.zeros(num_graphs, dtype=torch.long, device=batch.device)
    for idx in range(batch.size(0)):
        graph_idx = int(batch[idx].item())
        node_pos[idx] = graph_counts[graph_idx]
        graph_counts[graph_idx] += 1

    src, dst = edge_index
    for edge_idx in range(edge_index.size(1)):
        graph_idx = int(batch[src[edge_idx]].item())
        adjacency[graph_idx, node_pos[src[edge_idx]], node_pos[dst[edge_idx]]] = 1.0

    return adjacency


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        hidden_dim: int | None = None,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
