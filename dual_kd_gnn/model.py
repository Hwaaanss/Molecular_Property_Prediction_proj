from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINEConv, global_mean_pool


GNN_CONV_TYPES = ("gcn", "gine", "gat")

# Attention heads used by conv_type="gat". Fixed rather than tuned: the head
# count only reshuffles how hidden_dim is partitioned, and leaving it in the
# search space would spend trials on a knob that cannot change capacity.
GAT_HEADS = 4


class GNNEncoder(nn.Module):
    """Message-passing encoder for one view, with a selectable convolution.

    ``conv_type="gcn"`` (default) embeds the bond features and then collapses
    them to one scalar gate per edge, because ``GCNConv`` accepts only a scalar
    ``edge_weight``.

    ``conv_type="gine"`` removes that bottleneck. ``GINEConv`` adds the edge
    embedding to the neighbour features as a *vector* before the MLP, so the
    bond features survive at full width. That matters here more than it usually
    would: the last entry of the dual-branch bond feature is the interatomic
    distance, which is the only route by which 3D geometry reaches the message
    passing at all -- under GCN it is squeezed through a single sigmoid.

    ``conv_type="gat"`` weights each neighbour by learned attention, with the
    bond features entering the attention logit via ``edge_dim``. ``hidden_dim``
    is split across :data:`GAT_HEADS` heads and re-concatenated, so the output
    width matches the other two conv types exactly.

    The GINE MLP is ``Linear(in, hidden) -> ReLU -> Linear(hidden, hidden)``,
    deliberately narrower than the ``2*hidden`` bottleneck used by Hu et al.'s
    reference GIN, since these datasets run from 513 to 41k molecules.
    Normalization stays outside the MLP (``bn`` after the conv) so all three
    conv types share the identical residual/BN/dropout skeleton.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "gcn",
    ) -> None:
        super().__init__()
        if conv_type not in GNN_CONV_TYPES:
            raise ValueError(f"Unknown conv_type: {conv_type!r}. Expected one of {GNN_CONV_TYPES}.")
        self.conv_type = conv_type
        if conv_type == "gcn":
            self.edge_enc = nn.Linear(edge_dim, hidden_dim)
            self.edge_weight_proj = nn.Linear(hidden_dim, 1)
        else:
            # GINEConv and GATConv project edge_attr themselves (edge_dim=...),
            # per layer and to that layer's width, so the shared scalar gate has
            # no role.
            self.edge_enc = None
            self.edge_weight_proj = None
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = node_dim if layer_idx == 0 else hidden_dim
            self.convs.append(self._build_conv(conv_type, in_dim, hidden_dim, edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.res_projs.append(
                nn.Linear(in_dim, hidden_dim, bias=False) if in_dim != hidden_dim else nn.Identity()
            )
        self.dropout = dropout
        self.num_layers = num_layers

    @staticmethod
    def _build_conv(conv_type: str, in_dim: int, hidden_dim: int, edge_dim: int) -> nn.Module:
        if conv_type == "gcn":
            return GCNConv(in_dim, hidden_dim)
        if conv_type == "gine":
            return GINEConv(
                nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
                train_eps=True,
                edge_dim=edge_dim,
            )
        # gat: split hidden_dim across heads so the concatenated output is
        # exactly hidden_dim. A width that does not divide evenly falls back to
        # a single head rather than silently changing the layer's output size.
        heads = GAT_HEADS if hidden_dim % GAT_HEADS == 0 else 1
        return GATConv(
            in_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=True,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor):
        edge_weight = None
        if self.conv_type == "gcn" and edge_attr.size(0) > 0:
            edge_emb = self.edge_enc(edge_attr)
            edge_weight = torch.sigmoid(self.edge_weight_proj(edge_emb)).squeeze(-1)

        for layer_idx, (conv, bn, res_proj) in enumerate(zip(self.convs, self.bns, self.res_projs)):
            residual = res_proj(x)
            if self.conv_type == "gcn":
                message = conv(x, edge_index, edge_weight=edge_weight)
            else:
                message = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(message) + residual
            if layer_idx < self.num_layers - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class InteractionTensorHead(nn.Module):
    """Codebook interaction head: the logit is a pure quadratic form.

        logit_k(z) = z^T W_k z + b_k,   W_k = U_k V_k^T,   U_k = sum_m alpha_km C_m

    The only learned pieces are the shared prototypes ``C_m``, the per-task
    assignment logits ``theta_km`` that produce ``alpha_k``, and the per-task
    bias. There is deliberately **no** parallel linear term: an additive
    ``Linear(z)`` alongside the quadratic form is a second, separately-trained
    predictor, and whatever it explains is subtracted from what the interaction
    term has to explain -- which makes every block-norm statement about the
    quadratic form an understatement of unknown size. The bias stays because
    without it the form could not shift its output level at all.

    ``symmetric`` defaults to False, i.e. ``W_k = U_k V_k^T`` with independent
    factors, and that default is load-bearing rather than cosmetic. With
    ``symmetric=True`` the form is ``U_k U_k^T``, which is PSD, so
    ``z^T W_k z >= 0`` for every molecule. A task whose positive rate is 12%
    needs logits near -2, and the only signed quantity left after the linear
    term was removed is the per-task bias -- so gradient descent drives the
    codebook to zero and lets the bias fit the base rate. C = 0 is a stationary
    point of ``q = ||C^T z||^2`` (``dq/dC = 2 z z^T C``), so the head never
    recovers: every molecule gets the same logit and ROC-AUC lands on exactly
    0.5. Measured on ToxCast (617 tasks, median positive rate 0.125), 250 head
    steps: symmetric loss 0.6614 / logit spread 2.5e-3, asymmetric loss 0.1455 /
    logit spread 1.0. Regression fails the same way -- standardized targets are
    signed and a PSD form can only move up from the bias (FreeSolv RMSE 4.300
    symmetric vs 2.608 asymmetric, same architecture, 3 seeds).

    The routing function ``route`` is selected by ``assignment_mode``:
      - ``"hard"`` -> Gumbel-softmax with Straight-Through estimator
      - ``"soft"`` -> plain softmax
      - ``"sparse"`` -> top-k of softmax (renormalized)

    ``diversity_loss`` discourages codebook collapse by penalizing pairwise
    absolute cosine similarity between prototypes.

    When the codebook is switched off (``rank == 0`` or ``num_prototypes == 0``)
    the head degrades to a plain ``Linear(d, K)``. That is the "no codebook head"
    ablation: there is no meaningful half-way house, since prototypes and task
    weights are the head.

    Block structure
    ---------------
    ``block_dims`` declares how the input vector is partitioned across fusion
    branches (geometry, topology and -- when enabled -- fingerprint). The head
    keeps the block boundaries so :meth:`quadratic_block_norms` can report how
    much each block pair contributes to the learned quadratic form; the
    cross-block entries are what make the interaction claim checkable.

    ``block_proj=True`` makes the optional projection block-diagonal: each block
    is projected by its own ``Linear`` into its own slice of the output, so the
    boundaries survive the projection. The default (``False``) keeps a single
    ``Linear(d_model, proj_dim)``, which mixes all blocks -- under it the block
    decomposition is only exact when ``proj_dim == 0``.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        rank: int = 32,
        use_bias: bool = True,
        symmetric: bool = False,
        proj_dim: int = 0,
        num_prototypes: int = 6,
        assignment_mode: str = "hard",
        tau: float = 1.0,
        diversity_weight: float = 0.01,
        codebook_init: str = "orthogonal",
        codebook_init_scale: float = 0.1,
        topk: int = 2,
        block_dims: tuple[int, ...] | None = None,
        block_names: tuple[str, ...] | None = None,
        block_proj: bool = False,
        block_proj_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if assignment_mode not in {"hard", "soft", "sparse"}:
            raise ValueError(f"Unknown assignment_mode: {assignment_mode}")
        if codebook_init not in {"orthogonal", "random"}:
            raise ValueError(f"Unknown codebook_init: {codebook_init}")

        self.symmetric = symmetric
        self.num_classes = num_classes
        self.rank = int(rank)
        self.num_prototypes = int(num_prototypes)
        self.assignment_mode = assignment_mode
        self.diversity_weight = float(diversity_weight)
        self.topk = int(topk)
        self.tau = float(tau)

        self.block_dims = tuple(int(d) for d in block_dims) if block_dims else (int(d_model),)
        if sum(self.block_dims) != int(d_model):
            raise ValueError(
                f"block_dims {self.block_dims} sum to {sum(self.block_dims)}, "
                f"expected d_model={d_model}."
            )
        self.block_names = (
            tuple(block_names) if block_names is not None
            else tuple(f"b{i}" for i in range(len(self.block_dims)))
        )
        if len(self.block_names) != len(self.block_dims):
            raise ValueError("block_names and block_dims must have the same length.")
        self.block_proj = bool(block_proj) and proj_dim > 0

        if proj_dim > 0 and self.block_proj:
            # Block-diagonal projection: block i is projected by its own Linear
            # into its own output slice, so no weight ever couples two blocks and
            # the quadratic form's block decomposition stays exact.
            out_dims = (
                tuple(int(d) for d in block_proj_dims) if block_proj_dims is not None
                else self._split_projection_dims(self.block_dims, proj_dim)
            )
            if len(out_dims) != len(self.block_dims) or any(d < 1 for d in out_dims):
                raise ValueError(
                    f"block_proj_dims {out_dims} must give one positive width per block "
                    f"{self.block_dims}."
                )
            self.proj = None
            self.block_projs = nn.ModuleList(
                nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU())
                for d_in, d_out in zip(self.block_dims, out_dims)
            )
            effective_block_dims = out_dims
            effective_dim = sum(out_dims)
        elif proj_dim > 0:
            self.proj = nn.Sequential(nn.Linear(d_model, proj_dim), nn.ReLU())
            self.block_projs = None
            # The projection mixes every block, so the input partition does not
            # survive it; quadratic_block_norms refuses to report in this case.
            effective_block_dims = (proj_dim,)
            effective_dim = proj_dim
        else:
            self.proj = None
            self.block_projs = None
            effective_block_dims = self.block_dims
            effective_dim = d_model
        self.effective_dim = effective_dim
        self.effective_block_dims = tuple(effective_block_dims)
        offsets = [0]
        for width in self.effective_block_dims:
            offsets.append(offsets[-1] + width)
        # Start/end index of each block inside the (possibly projected) vector.
        self.block_slices = tuple(
            (offsets[i], offsets[i + 1]) for i in range(len(self.effective_block_dims))
        )
        # True when the block partition of the head input survives to the
        # quadratic form; False for a full (mixing) projection.
        self.blocks_preserved = (self.proj is None)

        # The codebook is the head. Losing either factor (rank or prototypes)
        # leaves nothing to build W_k from, so both fall back to the same plain
        # linear classifier rather than to a half-parameterized quadratic form.
        self._use_codebook = self.rank > 0 and self.num_prototypes > 0

        if self._use_codebook:
            self.codebook_u = nn.Parameter(
                self._init_codebook(self.num_prototypes, effective_dim, self.rank, codebook_init, codebook_init_scale)
            )
            self.assignment_logits_u = nn.Parameter(torch.zeros(num_classes, self.num_prototypes))
            nn.init.normal_(self.assignment_logits_u, std=1.0)
            if not symmetric:
                self.codebook_v = nn.Parameter(
                    self._init_codebook(self.num_prototypes, effective_dim, self.rank, codebook_init, codebook_init_scale)
                )
                self.assignment_logits_v = nn.Parameter(torch.zeros(num_classes, self.num_prototypes))
                nn.init.normal_(self.assignment_logits_v, std=1.0)
            self.linear_head = None
        else:
            self.linear_head = nn.Linear(effective_dim, num_classes, bias=False)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter("bias", None)

        # Device/dtype reference that survives every configuration above (the
        # head can legitimately own no weight matrix at all). Non-persistent, so
        # it never appears in a state_dict and old checkpoints still load.
        self.register_buffer("_anchor", torch.zeros(()), persistent=False)

    @staticmethod
    def _split_projection_dims(block_dims: tuple[int, ...], proj_dim: int) -> tuple[int, ...]:
        """Fallback split of ``proj_dim`` across blocks, proportional to width.

        Only used when the caller passes no explicit ``block_proj_dims``.
        DualDistillationModel always passes them, because the right split there
        is not proportional (see the note next to its call site).
        """
        total = sum(block_dims)
        widths = [max(1, int(round(proj_dim * d / total))) for d in block_dims]
        # Round-off lands on the widest block, which can absorb it.
        widest = max(range(len(widths)), key=lambda i: block_dims[i])
        widths[widest] = max(1, widths[widest] + proj_dim - sum(widths))
        return tuple(widths)

    @staticmethod
    def _init_codebook(num_prototypes: int, d: int, r: int, init: str, scale: float) -> torch.Tensor:
        codebook = torch.empty(num_prototypes, d, r)
        if init == "orthogonal":
            for m in range(num_prototypes):
                nn.init.orthogonal_(codebook[m])
            codebook.mul_(scale)
        else:
            codebook.normal_(0.0, 0.01)
        return codebook

    @property
    def use_codebook(self) -> bool:
        return self._use_codebook

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def _compute_assignment(self, logits: torch.Tensor) -> torch.Tensor:
        tau = max(self.tau, 1e-3)
        if self.training:
            if self.assignment_mode == "soft":
                return F.softmax(logits / tau, dim=-1)
            if self.assignment_mode == "hard":
                return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            k = min(self.topk, soft.size(-1))
            topk_vals, topk_idx = soft.topk(k, dim=-1)
            mask = torch.zeros_like(soft).scatter_(-1, topk_idx, topk_vals)
            return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.assignment_mode == "soft":
            return F.softmax(logits / tau, dim=-1)
        if self.assignment_mode == "sparse":
            soft = F.softmax(logits / tau, dim=-1)
            k = min(self.topk, soft.size(-1))
            topk_vals, topk_idx = soft.topk(k, dim=-1)
            mask = torch.zeros_like(soft).scatter_(-1, topk_idx, topk_vals)
            return mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=logits.size(-1)).to(logits.dtype)

    @staticmethod
    def _codebook_project(z: torch.Tensor, codebook: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # z: [B, d], codebook: [M, d, r], alpha: [K, M] -> [B, K, r]
        y = torch.einsum("bd,mdr->bmr", z, codebook)
        return torch.einsum("km,bmr->bkr", alpha, y)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the (optional) projection: full, block-diagonal, or none."""
        if self.proj is not None:
            return self.proj(z)
        if self.block_projs is None:
            return z
        parts = []
        start = 0
        for width, block_proj in zip(self.block_dims, self.block_projs):
            parts.append(block_proj(z[..., start:start + width]))
            start += width
        return torch.cat(parts, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.project(z)
        if self._use_codebook:
            alpha_u = self._compute_assignment(self.assignment_logits_u)
            u_proj = self._codebook_project(z, self.codebook_u, alpha_u)
            if self.symmetric:
                logits = (u_proj ** 2).sum(dim=-1)
            else:
                alpha_v = self._compute_assignment(self.assignment_logits_v)
                v_proj = self._codebook_project(z, self.codebook_v, alpha_v)
                logits = (u_proj * v_proj).sum(dim=-1)
        else:
            logits = self.linear_head(z)
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    @staticmethod
    def _pairwise_abs_cos(codebook: torch.Tensor) -> torch.Tensor:
        m = codebook.size(0)
        if m <= 1:
            return codebook.new_zeros(())
        flat = codebook.reshape(m, -1)
        normed = F.normalize(flat, dim=-1)
        sim = normed @ normed.t()
        off_diag = sim - torch.eye(m, device=sim.device, dtype=sim.dtype)
        return off_diag.abs().sum() / (m * (m - 1))

    def diversity_loss(self) -> torch.Tensor:
        if not self._use_codebook:
            return self._anchor.new_zeros(())
        loss = self._pairwise_abs_cos(self.codebook_u)
        if not self.symmetric:
            loss = 0.5 * (loss + self._pairwise_abs_cos(self.codebook_v))
        return loss

    @torch.no_grad()
    def quadratic_factors(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-class factors ``(U, V)`` of the quadratic form, or None if linear.

        Each has shape ``[K, effective_dim, rank]`` and the class-k quadratic
        form is ``W_k = U_k V_k^T`` (with ``V = U`` in the symmetric case), so
        the logit's quadratic part is ``z^T W_k z``. Codebook factors are
        resolved with the deterministic softmax assignment, not a Gumbel sample,
        so repeated calls on a fixed model agree.
        """
        if not self._use_codebook:
            return None
        tau = max(self.tau, 1e-3)
        alpha_u = F.softmax(self.assignment_logits_u / tau, dim=-1)
        factor_u = torch.einsum("km,mdr->kdr", alpha_u, self.codebook_u)
        if self.symmetric:
            return factor_u, factor_u
        alpha_v = F.softmax(self.assignment_logits_v / tau, dim=-1)
        return factor_u, torch.einsum("km,mdr->kdr", alpha_v, self.codebook_v)

    @torch.no_grad()
    def quadratic_block_norms(self, per_class: bool = False) -> dict[str, torch.Tensor]:
        """Frobenius norm of every block pair's contribution to the quadratic form.

        With blocks (geo, topo, fp) this returns the six entries
        ``geo-geo, topo-topo, fp-fp, geo-topo, geo-fp, topo-fp``; with the
        fingerprint branch off, the three entries over (geo, topo).

        For a diagonal pair the value is ``||W_ii||_F``. For a cross pair it is
        ``||W_ij + W_ji^T||_F``, i.e. the norm of the operator that actually
        multiplies ``z_i`` against ``z_j`` in ``z^T W z`` -- both off-diagonal
        halves contribute to that single bilinear term, so reporting only one of
        them would understate the coupling by a factor of two in the symmetric
        case.

        Returns one scalar tensor per pair, averaged over classes, or a ``[K]``
        tensor per pair when ``per_class=True``. Raises when a full (mixing)
        projection has destroyed the block partition.
        """
        if not self.blocks_preserved:
            raise RuntimeError(
                "Block decomposition is undefined: proj_dim > 0 with a full "
                "Linear(d_model, proj_dim) projection mixes every block. Rebuild the "
                "head with block_proj=True (--ih-block-proj) or with proj_dim=0."
            )

        names = self.block_names
        if len(names) != len(self.block_slices):
            names = tuple(f"b{i}" for i in range(len(self.block_slices)))

        factors = self.quadratic_factors()
        device = self._anchor.device
        dtype = self._anchor.dtype

        def reduce(value: torch.Tensor) -> torch.Tensor:
            return value if per_class else value.mean()

        result: dict[str, torch.Tensor] = {}
        zero = torch.zeros(self.num_classes, device=device, dtype=dtype)
        for i, (start_i, end_i) in enumerate(self.block_slices):
            for j, (start_j, end_j) in enumerate(self.block_slices):
                if j < i:
                    continue
                key = f"{names[i]}-{names[j]}"
                if factors is None:  # linear head: there is no quadratic term
                    result[key] = reduce(zero.clone())
                    continue
                u, v = factors
                u_i, v_i = u[:, start_i:end_i, :], v[:, start_i:end_i, :]
                u_j, v_j = u[:, start_j:end_j, :], v[:, start_j:end_j, :]
                block_ij = torch.einsum("kar,kbr->kab", u_i, v_j)
                if i == j:
                    contribution = block_ij
                else:
                    # W_ji^T = (U_j V_i^T)^T = V_i U_j^T
                    contribution = block_ij + torch.einsum("kar,kbr->kab", v_i, u_j)
                result[key] = reduce(contribution.flatten(1).norm(dim=-1))
        return result

    @torch.no_grad()
    def get_assignment_probabilities(self) -> torch.Tensor:
        """Return the soft assignment matrix alpha (K x M) for analysis/visualization."""
        if not self._use_codebook:
            return self._anchor.new_zeros((self.num_classes, 0))
        tau = max(self.tau, 1e-3)
        return F.softmax(self.assignment_logits_u / tau, dim=-1)


class DualDistillationModel(nn.Module):
    """Dual-view GNN with an EMA-teacher pretraining phase and a codebook head.

    Two phases, both of which used to live inside "stage 1"/"stage 2":

    Phase 1 (``forward_gcn_pretrain``) -- representation pretraining.
        Both view encoders train against the task loss, an EMA self-distillation
        MSE, and a cross-modal InfoNCE. The head is present (it is what turns the
        pooled vector into a task loss) and warm-starts here.

    Phase 2 (``forward``) -- head fine-tuning.
        The encoders are frozen and only the fusion norms, the fingerprint branch
        and the codebook head train. This is the phase whose learned quadratic
        form the block analysis reports on.

    There is no transformer stage: the per-view transformer encoders that used to
    sit between the frozen GNN and the head are gone, and with them the padded
    ``[B, L, H]`` node sequences they required. Pooling is now a scatter mean over
    the node tensor directly, which is numerically the same masked mean and much
    cheaper.
    """

    supports_stagewise_teacher = True

    def __init__(
        self,
        chem_dim: int = 19,
        phys_dim: int = 5,
        edge_dim: int = 7,
        gnn_hidden: int = 256,
        gnn_layers: int = 3,
        gnn_dropout: float = 0.4,
        gnn_conv: str = "gcn",
        fusion_dropout: float = 0.3,
        num_classes: int = 12,
        ih_rank: int = 32,
        ih_use_bias: bool = True,
        ih_symmetric: bool = False,
        ih_proj_dim: int = 0,
        ih_num_prototypes: int = 6,
        ih_assignment_mode: str = "hard",
        ih_tau_init: float = 1.0,
        ih_tau_final: float = 0.1,
        ih_diversity_weight: float = 0.01,
        ih_codebook_init: str = "orthogonal",
        ih_topk: int = 2,
        ih_block_proj: bool = False,
        info_nce_temperature: float = 0.2,
        zero_phys_branch: bool = False,
        use_fingerprint: bool = False,
        fp_input_dim: int | None = None,
        fp_dim: int = 64,
        fp_dropout: float = 0.2,
        fp_stage: str = "head",
        zero_fp_branch: bool = False,
    ) -> None:
        super().__init__()
        # "stage2" is the pre-refactor spelling of "head" and is still accepted
        # so saved configs and existing scripts keep working unchanged.
        if fp_stage == "stage2":
            fp_stage = "head"
        if fp_stage not in {"head", "both"}:
            raise ValueError(f"Unknown fp_stage: {fp_stage!r}. Expected 'head' ('stage2') or 'both'.")
        self.d_model = gnn_hidden
        self.ih_tau_init = float(ih_tau_init)
        self.ih_tau_final = float(ih_tau_final)
        self.info_nce_temperature = float(info_nce_temperature)
        self.zero_phys_branch = bool(zero_phys_branch)
        self.use_fingerprint = bool(use_fingerprint)
        self.zero_fp_branch = bool(zero_fp_branch)
        self.fp_stage = fp_stage
        self.fp_dim = int(fp_dim) if self.use_fingerprint else 0
        if self.use_fingerprint and fp_input_dim is None:
            # Imported lazily so `import model` stays free of the RDKit stack.
            from common.data import DEFAULT_FINGERPRINT_DIM

            fp_input_dim = DEFAULT_FINGERPRINT_DIM
        self.fp_input_dim = int(fp_input_dim) if fp_input_dim is not None else 0

        self.gnn_conv = gnn_conv
        self.gnn_c = GNNEncoder(chem_dim, edge_dim, gnn_hidden, gnn_layers, gnn_dropout, gnn_conv)
        self.gnn_p = GNNEncoder(phys_dim, edge_dim, gnn_hidden, gnn_layers, gnn_dropout, gnn_conv)

        self.teacher_gnn_c = copy.deepcopy(self.gnn_c)
        self.teacher_gnn_p = copy.deepcopy(self.gnn_p)
        self._freeze_teacher_parameters()
        self.set_teacher_eval()

        self.input_norm_c = nn.LayerNorm(gnn_hidden)
        self.input_norm_p = nn.LayerNorm(gnn_hidden)
        self.input_drop = nn.Dropout(fusion_dropout)

        # Fusion order is fixed: geometry (physical), topology (chemical), then
        # fingerprint. The first two must stay in this order -- the existing
        # cat([phys, chem]) is what every saved checkpoint and every published
        # block-structure claim assumes.
        self.block_names = ("geo", "topo") + (("fp",) if self.use_fingerprint else ())
        self.block_dims = (gnn_hidden, gnn_hidden) + ((self.fp_dim,) if self.use_fingerprint else ())
        self.concat_dim = sum(self.block_dims)

        if self.use_fingerprint:
            # Per-block LayerNorm. A single LayerNorm over the whole fused vector
            # would mix the blocks' statistics, so the zero-filled fingerprint
            # slot used in phase 1 would shift the mean and variance seen by the
            # geometry and topology halves -- i.e. enabling the branch would
            # change phase 1 even though the branch contributes nothing there.
            # Same reasoning as the block-diagonal projection below.
            self.concat_norm = None
            self.fuse_norm_geo = nn.LayerNorm(gnn_hidden)
            self.fuse_norm_topo = nn.LayerNorm(gnn_hidden)
        else:
            # One LayerNorm over [geo, topo], same module name and parameter
            # shapes as before the fingerprint branch existed.
            self.concat_norm = nn.LayerNorm(self.concat_dim)

        # Per-block projection widths for the block-diagonal head projection.
        # The two node blocks are the same width (gnn_hidden each) and split
        # ih_proj_dim evenly. The fingerprint block keeps its own width instead
        # of taking a proportional share: fp_dim is already this branch's
        # bottleneck (fp_input_dim -> fp_dim), and a proportional share would
        # crush it to single digits under typical tuned configs (BACE:
        # 128 * 64/1088 = 8 dims), erasing the very cross block the branch
        # exists to measure.
        block_proj_dims = None
        if ih_block_proj and ih_proj_dim > 0:
            geo_width = ih_proj_dim // 2
            block_proj_dims = (geo_width, ih_proj_dim - geo_width)
            if self.use_fingerprint:
                block_proj_dims = block_proj_dims + (self.fp_dim,)

        self.classifier = InteractionTensorHead(
            d_model=self.concat_dim,
            num_classes=num_classes,
            rank=ih_rank,
            use_bias=ih_use_bias,
            symmetric=ih_symmetric,
            proj_dim=ih_proj_dim,
            num_prototypes=ih_num_prototypes,
            assignment_mode=ih_assignment_mode,
            tau=ih_tau_init,
            diversity_weight=ih_diversity_weight,
            codebook_init=ih_codebook_init,
            topk=ih_topk,
            block_dims=self.block_dims,
            block_names=self.block_names,
            block_proj=ih_block_proj,
            block_proj_dims=block_proj_dims,
        )

        # Asymmetric predictors for cross-modal InfoNCE distillation
        # (CLIP-style). Applied on graph-pooled student GNN outputs;
        # EMA teacher (opposite branch) graph pool is the positive key,
        # other graphs in the batch are negatives. Stop-grad on keys.
        self.predictor_c2p = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.BatchNorm1d(gnn_hidden),
            nn.GELU(),
            nn.Linear(gnn_hidden, gnn_hidden),
        )
        self.predictor_p2c = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.BatchNorm1d(gnn_hidden),
            nn.GELU(),
            nn.Linear(gnn_hidden, gnn_hidden),
        )

        # Built LAST, on purpose. Every module above draws from the global RNG,
        # so creating the fingerprint encoder earlier would shift the random
        # stream and give the fingerprint-on and fingerprint-off runs different
        # GNN initializations at the same seed, making their phase-1
        # trajectories incomparable. Kept here, everything up to the classifier
        # initializes identically in both conditions.
        #
        # Deliberately narrow: fp_input_dim is >2000 while the small datasets are
        # ~1.5k molecules, so this is the first place to overfit. fp_dim defaults
        # to 64 and the block stays that width all the way into the head.
        if self.use_fingerprint:
            self.fp_encoder = nn.Sequential(
                nn.Linear(self.fp_input_dim, self.fp_dim),
                nn.LayerNorm(self.fp_dim),
                nn.GELU(),
                nn.Dropout(fp_dropout),
                nn.Linear(self.fp_dim, self.fp_dim),
            )
            # The fingerprint block's own fusion LayerNorm, the third sibling of
            # fuse_norm_geo / fuse_norm_topo. Never applied in phase 1 under the
            # default fp_stage: that path substitutes an exact zero vector
            # instead of calling the branch, so these parameters receive no
            # gradient before head fine-tuning.
            self.fp_norm = nn.LayerNorm(self.fp_dim)
        else:
            self.fp_encoder = None
            self.fp_norm = None

        self.sync_teachers()

    # ------------------------------------------------------------------ teacher

    def _freeze_teacher_parameters(self) -> None:
        for param in self.teacher_gnn_c.parameters():
            param.requires_grad = False
        for param in self.teacher_gnn_p.parameters():
            param.requires_grad = False

    def set_teacher_eval(self) -> None:
        self.teacher_gnn_c.eval()
        self.teacher_gnn_p.eval()

    @torch.no_grad()
    def sync_teachers(self) -> None:
        self.teacher_gnn_c.load_state_dict(self.gnn_c.state_dict())
        self.teacher_gnn_p.load_state_dict(self.gnn_p.state_dict())
        self._freeze_teacher_parameters()
        self.set_teacher_eval()

    @torch.no_grad()
    def _ema_update_module(self, teacher: nn.Module, student: nn.Module, ema_decay: float) -> None:
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1.0 - ema_decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.data.mul_(ema_decay).add_(student_buffer.data, alpha=1.0 - ema_decay)
            else:
                teacher_buffer.data.copy_(student_buffer.data)

    @torch.no_grad()
    def update_teachers(self, ema_decay: float = 0.99) -> None:
        self._ema_update_module(self.teacher_gnn_c, self.gnn_c, ema_decay)
        self._ema_update_module(self.teacher_gnn_p, self.gnn_p, ema_decay)
        self.set_teacher_eval()

    # ------------------------------------------------------------- regularizers

    def auxiliary_loss(self) -> torch.Tensor:
        """Diversity regularization on the classifier codebook (zero when disabled)."""
        if not self.classifier.use_codebook or self.classifier.diversity_weight <= 0.0:
            return self.classifier._anchor.new_zeros(())
        return self.classifier.diversity_weight * self.classifier.diversity_loss()

    def step_classifier_schedule(self, stage: str, epoch_idx: int, total_epochs: int) -> None:
        """Anneal classifier sampling temperature linearly over head-phase epochs."""
        if stage not in {"head", "stage2"} or not self.classifier.use_codebook:
            return
        if total_epochs <= 1:
            self.classifier.set_tau(self.ih_tau_final)
            return
        progress = max(0.0, min(1.0, float(epoch_idx) / float(total_epochs - 1)))
        tau = self.ih_tau_init + (self.ih_tau_final - self.ih_tau_init) * progress
        self.classifier.set_tau(tau)

    # -------------------------------------------------------------- param groups

    def _fusion_norm_parameters(self):
        """Parameters of the fusion LayerNorm(s), whichever variant is in use."""
        if self.concat_norm is not None:
            return list(self.concat_norm.parameters())
        return list(self.fuse_norm_geo.parameters()) + list(self.fuse_norm_topo.parameters())

    def _fingerprint_parameters(self):
        """Fingerprint encoder + its fusion LayerNorm, as one group."""
        if not self.use_fingerprint:
            return []
        return list(self.fp_encoder.parameters()) + list(self.fp_norm.parameters())

    def get_gcn_pretrain_parameters(self):
        params = []
        params += list(self.gnn_c.parameters())
        params += list(self.gnn_p.parameters())
        params += list(self.input_norm_c.parameters())
        params += list(self.input_norm_p.parameters())
        params += self._fusion_norm_parameters()
        params += list(self.classifier.parameters())
        params += list(self.predictor_c2p.parameters())
        params += list(self.predictor_p2c.parameters())
        # Excluded unless fp_stage == "both": with the default the fingerprint
        # branch must not train during encoder pretraining, otherwise the head
        # learns to read labels off the fingerprint and the gradient reaching the
        # GNN encoders weakens -- and those encoders are frozen in phase 2, so
        # anything they fail to learn in phase 1 can never be recovered.
        if self.fp_stage == "both":
            params += self._fingerprint_parameters()
        return params

    def get_gcn_pretrain_param_groups(self, default_weight_decay: float, predictor_weight_decay: float = 1e-4):
        main_params = []
        main_params += list(self.gnn_c.parameters())
        main_params += list(self.gnn_p.parameters())
        main_params += list(self.input_norm_c.parameters())
        main_params += list(self.input_norm_p.parameters())
        main_params += self._fusion_norm_parameters()
        main_params += list(self.classifier.parameters())
        # See get_gcn_pretrain_parameters: phase-1 membership is opt-in.
        # Weight decay: the fingerprint encoder rides in the main group at the
        # run's default weight decay rather than getting a predictor-style group
        # of its own. The overfitting risk that would address is already handled
        # structurally by the narrow fp_dim (64) plus fp_dropout.
        if self.fp_stage == "both":
            main_params += self._fingerprint_parameters()
        predictor_params = []
        predictor_params += list(self.predictor_c2p.parameters())
        predictor_params += list(self.predictor_p2c.parameters())
        return [
            {"params": main_params, "weight_decay": float(default_weight_decay)},
            {"params": predictor_params, "weight_decay": float(predictor_weight_decay)},
        ]

    def get_head_parameters(self):
        """Everything trained in phase 2. The GNN encoders are deliberately absent."""
        params = []
        params += list(self.input_norm_c.parameters())
        params += list(self.input_norm_p.parameters())
        params += self._fusion_norm_parameters()
        params += list(self.classifier.parameters())
        params += self._fingerprint_parameters()
        return params

    # Pre-refactor name, kept so external scripts and the Trainer's older call
    # sites keep resolving. There is no transformer left to train.
    get_transformer_parameters = get_head_parameters

    # ------------------------------------------------------------------ encoding

    def _run_student_gnn(self, data):
        chem_nodes = self.gnn_c(data.x_chem, data.edge_index, data.edge_attr, data.batch)
        x_phys = torch.zeros_like(data.x_phys) if self.zero_phys_branch else data.x_phys
        phys_nodes = self.gnn_p(x_phys, data.edge_index, data.edge_attr, data.batch)
        return chem_nodes, phys_nodes

    @torch.no_grad()
    def _run_frozen_student_gnn(self, data):
        prev_chem_training = self.gnn_c.training
        prev_phys_training = self.gnn_p.training
        self.gnn_c.eval()
        self.gnn_p.eval()
        try:
            return self._run_student_gnn(data)
        finally:
            self.gnn_c.train(prev_chem_training)
            self.gnn_p.train(prev_phys_training)

    @torch.no_grad()
    def _run_teacher_gnn(self, data):
        self.set_teacher_eval()
        chem_nodes = self.teacher_gnn_c(data.x_chem, data.edge_index, data.edge_attr, data.batch)
        x_phys = torch.zeros_like(data.x_phys) if self.zero_phys_branch else data.x_phys
        phys_nodes = self.teacher_gnn_p(x_phys, data.edge_index, data.edge_attr, data.batch)
        return chem_nodes, phys_nodes

    @staticmethod
    def _num_graphs(data) -> int:
        return int(data.batch.max().item()) + 1

    def apply_fusion_norm(self, fused_nodes: torch.Tensor) -> torch.Tensor:
        """Normalize the fused node features [N, geo+topo].

        One LayerNorm over the whole vector when the fingerprint branch is off,
        otherwise an independent LayerNorm per block. The fingerprint block is
        not part of this tensor at all: it has no node dimension and is appended
        after pooling, with its own LayerNorm.
        """
        if self.concat_norm is not None:
            return self.concat_norm(fused_nodes)
        geo, topo = fused_nodes.split(self.d_model, dim=-1)
        return torch.cat([self.fuse_norm_geo(geo), self.fuse_norm_topo(topo)], dim=-1)

    def encode_fingerprint(self, data) -> torch.Tensor:
        """Fingerprint branch output [B, fp_dim] for the pooled fusion vector."""
        fp = getattr(data, "fp", None)
        if fp is None:
            raise ValueError(
                "use_fingerprint=True but the batch carries no `fp` attribute. Rebuild the "
                "datasets with the current common/data.py (and refresh the feature cache: "
                "`python scripts/precompute_features.py --datasets <name>`)."
            )
        if fp.dim() != 2 or fp.size(-1) != self.fp_input_dim:
            raise ValueError(
                f"data.fp has shape {tuple(fp.shape)}, expected [B, {self.fp_input_dim}]. "
                "Datasets must store it as [1, F] per molecule so PyG batches it to [B, F]."
            )
        fp = fp.to(dtype=self.fp_encoder[0].weight.dtype)
        if self.zero_fp_branch:
            fp = torch.zeros_like(fp)
        return self.fp_norm(self.fp_encoder(fp))

    def _append_fingerprint(self, graph_repr: torch.Tensor, data, encode: bool) -> torch.Tensor:
        """Concatenate the fingerprint block onto the pooled [geo, topo] vector.

        ``encode=False`` appends an exact zero vector instead of running the
        branch. That is the phase-1 default: the fp slot exists from the start so
        the single warm-started classifier keeps one shape across both phases,
        but nothing flows through it, so the codebook rows sitting on that slot
        receive no gradient and stay at their initial values until phase 2.
        """
        if not self.use_fingerprint:
            return graph_repr
        if encode:
            fp_repr = self.encode_fingerprint(data)
        else:
            fp_repr = graph_repr.new_zeros(graph_repr.size(0), self.fp_dim)
        return torch.cat([graph_repr, fp_repr], dim=-1)

    def _fuse_and_pool(self, chem_nodes, phys_nodes, data, encode_fp: bool) -> torch.Tensor:
        """[geo, topo] per node -> normalize -> mean pool -> append fingerprint."""
        chem = self.input_drop(self.input_norm_c(chem_nodes))
        phys = self.input_drop(self.input_norm_p(phys_nodes))
        fused = torch.cat([phys, chem], dim=-1)
        fused = self.apply_fusion_norm(fused)
        graph_repr = global_mean_pool(fused, data.batch, size=self._num_graphs(data))
        return self._append_fingerprint(graph_repr, data, encode=encode_fp)

    # -------------------------------------------------------------------- phase 1

    def forward_gcn_pretrain(self, data):
        teacher_edge_dropout = 0.0
        student_edge_dropout = 0.1

        student_data = copy.copy(data)
        student_edge_index = data.edge_index.clone()
        student_edge_attr = data.edge_attr.clone() if getattr(data, "edge_attr", None) is not None else None
        dropout_before_edges = int(student_edge_index.size(1))

        if dropout_before_edges > 0:
            keep_mask = torch.rand(dropout_before_edges, device=student_edge_index.device) >= student_edge_dropout
            student_edge_index = student_edge_index[:, keep_mask]
            if student_edge_attr is not None and student_edge_attr.size(0) == dropout_before_edges:
                student_edge_attr = student_edge_attr[keep_mask]

        student_data.edge_index = student_edge_index
        if student_edge_attr is not None:
            student_data.edge_attr = student_edge_attr

        teacher_chem, teacher_phys = self._run_teacher_gnn(data)
        student_chem, student_phys = self._run_student_gnn(student_data)

        # Phase 1 keeps the fingerprint slot zero-filled unless fp_stage="both".
        student_graph_repr = self._fuse_and_pool(
            student_chem, student_phys, data, encode_fp=self.fp_stage == "both"
        )
        student_logits = self.classifier(student_graph_repr)

        # Graph-level pools for cross-modal InfoNCE, on the raw encoder outputs.
        num_graphs = self._num_graphs(data)
        student_c_graph = global_mean_pool(student_chem, data.batch, size=num_graphs)
        student_p_graph = global_mean_pool(student_phys, data.batch, size=num_graphs)
        teacher_c_graph = global_mean_pool(teacher_chem, data.batch, size=num_graphs)
        teacher_p_graph = global_mean_pool(teacher_phys, data.batch, size=num_graphs)

        # Asymmetric predictors at graph level. Targets (EMA teacher) detached.
        q_c_to_p = self.predictor_c2p(student_c_graph)
        q_p_to_c = self.predictor_p2c(student_p_graph)

        return {
            # Node-level [N, H] tensors. The padded [B, L, H] sequences the
            # transformer stage needed are gone; the KD MSE and the graph pools
            # below are identical either way.
            "student_phys_seq": student_phys,
            "student_chem_seq": student_chem,
            "student_logits": student_logits,
            "student_chem_to_phys_graph": q_c_to_p,
            "student_phys_to_chem_graph": q_p_to_c,
            "teacher_phys_seq": teacher_phys.detach(),
            "teacher_chem_seq": teacher_chem.detach(),
            "teacher_phys_graph": teacher_p_graph.detach(),
            "teacher_chem_graph": teacher_c_graph.detach(),
            "batch": data.batch,
            "pad_mask": None,
            "debug_info": {
                "teacher_edge_dropout": teacher_edge_dropout,
                "student_edge_dropout": student_edge_dropout,
                "teacher_num_edges": int(data.edge_index.size(1)),
                "student_num_edges": int(student_data.edge_index.size(1)),
                "dropout_before_edges": dropout_before_edges,
                "dropout_after_edges": int(student_data.edge_index.size(1)),
                "teacher_edge_index_shape": tuple(data.edge_index.shape),
                "student_edge_index_shape": tuple(student_data.edge_index.shape),
                "student_uses_separate_edge_tensor": student_data.edge_index.data_ptr() != data.edge_index.data_ptr(),
            },
        }

    def compute_distill_loss(self, stage_out):
        loss_phys = F.mse_loss(stage_out["student_phys_seq"], stage_out["teacher_phys_seq"])
        loss_chem = F.mse_loss(stage_out["student_chem_seq"], stage_out["teacher_chem_seq"])
        return 0.5 * (loss_phys + loss_chem)

    def compute_cross_distill_loss(self, stage_out):
        # Symmetric InfoNCE on graph-pooled representations. The student
        # predictor output is the query; the opposite-branch EMA teacher pool
        # (already detached) is the positive key; other graphs in the batch
        # are negatives. Lower bound on I(chem; phys) per van den Oord 2018.
        q_c2p = stage_out["student_chem_to_phys_graph"]
        q_p2c = stage_out["student_phys_to_chem_graph"]
        k_p = stage_out["teacher_phys_graph"]
        k_c = stage_out["teacher_chem_graph"]

        batch_size = q_c2p.size(0)
        if batch_size < 2:
            return q_c2p.new_zeros(())

        tau = max(self.info_nce_temperature, 1e-3)
        q_c2p = F.normalize(q_c2p, dim=-1, eps=1e-8)
        q_p2c = F.normalize(q_p2c, dim=-1, eps=1e-8)
        k_p = F.normalize(k_p, dim=-1, eps=1e-8)
        k_c = F.normalize(k_c, dim=-1, eps=1e-8)

        logits_c2p = (q_c2p @ k_p.t()) / tau
        logits_p2c = (q_p2c @ k_c.t()) / tau
        labels = torch.arange(batch_size, device=q_c2p.device)

        loss_c2p = F.cross_entropy(logits_c2p, labels)
        loss_p2c = F.cross_entropy(logits_p2c, labels)
        return 0.5 * (loss_c2p + loss_p2c)

    # -------------------------------------------------------------------- phase 2

    def fused_graph_representation(self, data) -> torch.Tensor:
        """Phase-2 pooled fusion vector [B, concat_dim], i.e. the head's input.

        The encoders run under ``no_grad`` in eval mode: phase 2 fine-tunes the
        head on fixed representations. Exposed so analysis code reads the real
        fusion vector instead of re-deriving it (and drifting from it).
        """
        student_chem, student_phys = self._run_frozen_student_gnn(data)
        # The fingerprint branch is live here regardless of fp_stage.
        return self._fuse_and_pool(student_chem, student_phys, data, encode_fp=True)

    def forward(self, data):
        return self.classifier(self.fused_graph_representation(data))
