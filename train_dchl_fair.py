import argparse
import importlib.util
import os
import os.path as osp
import sys
import time

import numpy as np

import utils as eagle_utils
from utils import (
    add_metric_sums,
    collect_eval_batch,
    compute_ranking_metric_sums,
    describe_loaded_data,
    finalize_metric_sums,
    load_datasets,
    save_config,
    save_metrics,
    set_random_seed,
)


REPO_DIR = osp.dirname(osp.abspath(__file__))
DCHL_DIR = osp.join(REPO_DIR, "baseline_DCHL")


def ensure_dchl_import_path():
    if not osp.isdir(DCHL_DIR):
        raise FileNotFoundError(f"DCHL directory not found: {DCHL_DIR}")
    if DCHL_DIR in sys.path:
        sys.path.remove(DCHL_DIR)
    sys.path.insert(0, DCHL_DIR)


def import_dchl():
    ensure_dchl_import_path()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import scipy.sparse as sp

    model_path = osp.join(DCHL_DIR, "model.py")
    spec = importlib.util.spec_from_file_location("baseline_dchl_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import DCHL model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DCHL = module.DCHL

    return torch, nn, F, sp, DCHL


def sync_device(torch, device):
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda_peaks(torch, device):
    if getattr(device, "type", None) != "cuda":
        return False
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    if hasattr(torch.cuda, "reset_max_memory_allocated"):
        torch.cuda.reset_max_memory_allocated(device)
    if hasattr(torch.cuda, "reset_max_memory_reserved"):
        torch.cuda.reset_max_memory_reserved(device)
    return True


def cuda_peak_allocated(torch, device):
    if getattr(device, "type", None) != "cuda":
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def cuda_peak_reserved(torch, device):
    if getattr(device, "type", None) != "cuda":
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_reserved(device))


def format_bytes(value):
    if value is None:
        return "n/a"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}GiB"


def snapshots_to_events4(snapshot_list):
    chunks = []
    for events, t_norm, _ in snapshot_list:
        if len(events) == 0:
            continue
        t_col = np.full((len(events), 1), int(t_norm), dtype=np.int64)
        chunks.append(np.hstack((events.astype(np.int64, copy=False), t_col)))
    if not chunks:
        return np.empty((0, 4), dtype=np.int64)
    return np.vstack(chunks).astype(np.int64, copy=False)


def snapshot_raw_times(snapshot_list):
    return [int(t_orig) for _, _, t_orig in snapshot_list]


def normalize_adj(sp, adj, is_symmetric=False):
    rowsum = np.asarray(adj.sum(1)).reshape(-1)
    if is_symmetric:
        d_inv = np.power(rowsum + 1e-8, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        return d_mat @ adj @ d_mat
    d_inv = np.power(rowsum + 1e-8, -1.0)
    d_inv[np.isinf(d_inv)] = 0.0
    return sp.diags(d_inv) @ adj


def row_normalize_incidence(sp, incidence):
    rowsum = np.asarray(incidence.sum(1)).reshape(-1)
    d_inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum > 0
    d_inv[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(d_inv) @ incidence


def drop_csr_edges(sp, matrix, keep_rate):
    keep_rate = float(keep_rate)
    if keep_rate >= 1.0:
        return matrix.tocsr()
    coo = matrix.tocoo()
    if coo.nnz == 0:
        return matrix.tocsr()
    mask = np.floor(np.random.rand(coo.nnz) + keep_rate).astype(bool)
    return sp.csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])),
        shape=coo.shape,
        dtype=np.float32,
    )


def csr_to_torch_sparse(torch, csr, device):
    coo = csr.tocoo()
    idx = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    indices = torch.from_numpy(idx).long()
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    tensor = torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32)
    return tensor.coalesce().to(device)


def local_business_ids(events4, business_first):
    return (events4[:, 2].astype(np.int64, copy=False) - int(business_first)).astype(np.int64, copy=False)


def build_user_sequences(train_events4, num_users, business_first):
    seqs = [[] for _ in range(int(num_users))]
    if len(train_events4) == 0:
        return seqs
    order = np.argsort(train_events4[:, 3], kind="stable")
    for s, _, o, _ in train_events4[order]:
        seqs[int(s)].append(int(o) - int(business_first))
    return seqs


def build_user_poi_hypergraph(sp, train_events4, num_users, num_pois, business_first, friend_edges, args):
    if len(train_events4) == 0:
        return sp.csr_matrix((num_pois, num_users), dtype=np.float32)

    users = train_events4[:, 0].astype(np.int64, copy=False)
    pois = local_business_ids(train_events4, business_first)
    valid = (users >= 0) & (users < int(num_users)) & (pois >= 0) & (pois < int(num_pois))
    data = np.ones(int(valid.sum()), dtype=np.float32)
    own = sp.csr_matrix((data, (pois[valid], users[valid])), shape=(num_pois, num_users), dtype=np.float32)
    own.data[:] = 1.0

    if not bool(args.use_friend_edges) or friend_edges is None or len(friend_edges) == 0 or float(args.friend_weight) <= 0:
        return drop_csr_edges(sp, own, args.keep_rate)

    src = friend_edges[:, 0].astype(np.int64, copy=False)
    dst = friend_edges[:, 1].astype(np.int64, copy=False)
    mask = (src >= 0) & (src < int(num_users)) & (dst >= 0) & (dst < int(num_users))
    src = src[mask]
    dst = dst[mask]
    if bool(args.friend_undirected):
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    values = np.ones(len(src), dtype=np.float32)
    friend_adj = sp.csr_matrix((values, (src, dst)), shape=(num_users, num_users), dtype=np.float32)
    friend_adj.data[:] = 1.0
    friend_adj = normalize_adj(sp, friend_adj, is_symmetric=False)
    friend_pois = own @ friend_adj.T
    combined = own + float(args.friend_weight) * friend_pois
    combined.data = np.clip(combined.data, 0.0, 1.0 + float(args.friend_weight))
    return drop_csr_edges(sp, combined, args.keep_rate)


def build_directed_poi_graph(sp, user_seqs, num_pois, window, keep_rate):
    rows = []
    cols = []
    max_window = int(window)
    for seq in user_seqs:
        n = len(seq)
        if n < 2:
            continue
        for i, src in enumerate(seq[:-1]):
            end = n if max_window <= 0 else min(n, i + max_window + 1)
            for j in range(i + 1, end):
                rows.append(int(src))
                cols.append(int(seq[j]))
    if not rows:
        return sp.csr_matrix((num_pois, num_pois), dtype=np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(num_pois, num_pois), dtype=np.float32)
    mat.data[:] = 1.0
    return drop_csr_edges(sp, mat, keep_rate)


def build_geo_graph(sp, data, args):
    num_pois = int(data["num_businesses"])
    lat = np.asarray(data["business_latitude"], dtype=np.float64)
    lon = np.asarray(data["business_longitude"], dtype=np.float64)
    if len(lat) != num_pois or len(lon) != num_pois:
        return sp.eye(num_pois, format="csr", dtype=np.float32)

    threshold = float(args.distance_threshold)
    earth_km = 6371.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    rows = []
    cols = []
    chunk_size = max(1, int(args.geo_chunk_size))
    for start in range(0, num_pois, chunk_size):
        end = min(num_pois, start + chunk_size)
        dlat = lat_rad[start:end, None] - lat_rad[None, :]
        dlon = lon_rad[start:end, None] - lon_rad[None, :]
        a = np.sin(dlat * 0.5) ** 2 + np.cos(lat_rad[start:end, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon * 0.5) ** 2
        dist = 2.0 * earth_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        rr, cc = np.where(dist <= threshold)
        rows.append(rr.astype(np.int64) + start)
        cols.append(cc.astype(np.int64))
    if not rows:
        return sp.eye(num_pois, format="csr", dtype=np.float32)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    values = np.ones(len(row), dtype=np.float32)
    geo = sp.csr_matrix((values, (row, col)), shape=(num_pois, num_pois), dtype=np.float32)
    geo = geo.maximum(sp.eye(num_pois, format="csr", dtype=np.float32))
    return normalize_adj(sp, geo, is_symmetric=False).tocsr()


class THGGraphForDCHL:
    def __init__(self, torch, sp, data, train_events4, args, device):
        self.num_users = int(data["num_users"])
        self.num_pois = int(data["num_businesses"])
        self.business_first = int(data["business_first_id"])
        self.padding_idx = self.num_pois
        self.device = device

        user_seqs = build_user_sequences(train_events4, self.num_users, self.business_first)
        self.user_seqs = user_seqs
        max_len = max((len(seq) for seq in user_seqs), default=0)
        max_len = min(max_len, max(0, int(args.max_seq_len)))
        if max_len > 0:
            padded = np.full((self.num_users, max_len), self.padding_idx, dtype=np.int64)
            for u, seq in enumerate(user_seqs):
                if seq:
                    tail = seq[-max_len:]
                    padded[u, : len(tail)] = np.asarray(tail, dtype=np.int64)
            self.pad_all_train_sessions = torch.from_numpy(padded).long().to(device)
        else:
            self.pad_all_train_sessions = torch.empty((self.num_users, 0), dtype=torch.long, device=device)

        h_pu = build_user_poi_hypergraph(
            sp,
            train_events4,
            self.num_users,
            self.num_pois,
            self.business_first,
            data.get("static_user_friend_edges"),
            args,
        )
        self.H_pu = h_pu
        self.HG_pu = csr_to_torch_sparse(torch, row_normalize_incidence(sp, h_pu), device)
        h_up = h_pu.T.tocsr()
        self.HG_up = csr_to_torch_sparse(torch, row_normalize_incidence(sp, h_up), device)

        geo = build_geo_graph(sp, data, args)
        self.poi_geo_graph = csr_to_torch_sparse(torch, geo, device)

        h_src = build_directed_poi_graph(sp, user_seqs, self.num_pois, args.transition_window, args.keep_rate_poi)
        h_tar = h_src.T.tocsr()
        self.HG_poi_src = csr_to_torch_sparse(torch, row_normalize_incidence(sp, h_src), device)
        self.HG_poi_tar = csr_to_torch_sparse(torch, row_normalize_incidence(sp, h_tar), device)


class THGEventDataset:
    def __init__(self, events4, business_first, num_pois, max_time_norm, num_time_bins, device):
        self.events4 = np.asarray(events4, dtype=np.int64)
        self.business_first = int(business_first)
        self.num_pois = int(num_pois)
        self.max_time_norm = max(0, int(max_time_norm))
        self.num_time_bins = max(1, int(num_time_bins))
        self.device = device

    def __len__(self):
        return int(len(self.events4))

    def __getitem__(self, idx):
        s, r, o, t = self.events4[int(idx)]
        label = int(o) - self.business_first
        return int(s), int(r), int(t), int(label)


def collate_thg_events(torch, device, max_time_norm, num_time_bins, batch):
    arr = np.asarray(batch, dtype=np.int64)
    t = arr[:, 2]
    denom = max(int(max_time_norm) + 1, 1)
    time_bin = np.floor(t.astype(np.float64) * int(num_time_bins) / denom).astype(np.int64)
    time_bin = np.clip(time_bin, 0, int(num_time_bins) - 1)
    return {
        "user_idx": torch.from_numpy(arr[:, 0]).long().to(device),
        "rel_idx": torch.from_numpy(arr[:, 1]).long().to(device),
        "time_idx": torch.from_numpy(t).long().to(device),
        "time_bin": torch.from_numpy(time_bin).long().to(device),
        "label": torch.from_numpy(arr[:, 3]).long().to(device),
    }


def make_thg_dchl_model_class(torch, nn, F, DCHL):
    class _THGDCHLModel(nn.Module):
        def __init__(self, num_users, num_pois, num_rels, max_time_norm, args, device):
            super().__init__()
            self.args = args
            self.device = device
            self.num_rels = int(num_rels)
            self.max_time_norm = int(max_time_norm)
            self.num_time_bins = int(args.num_time_bins)
            self.dchl = DCHL(num_users, num_pois, args, device)
            self.rel_embedding = nn.Embedding(num_rels, args.emb_dim)
            self.time_embedding = nn.Embedding(self.num_time_bins, args.emb_dim)
            self.query_layer_norm = nn.LayerNorm(args.emb_dim)
            nn.init.xavier_uniform_(self.rel_embedding.weight)
            nn.init.xavier_uniform_(self.time_embedding.weight)

        def _base_forward(self, dataset, batch, compute_cl):
            model = self.dchl
            saved_dropouts = None
            if not self.training:
                saved_dropouts = (
                    model.mv_hconv_network.dropout,
                    model.di_hconv_network.dropout,
                )
                model.mv_hconv_network.dropout = 0.0
                model.di_hconv_network.dropout = 0.0
            try:
                return self._base_forward_impl(model, dataset, batch, compute_cl)
            finally:
                if saved_dropouts is not None:
                    model.mv_hconv_network.dropout, model.di_hconv_network.dropout = saved_dropouts

        def _base_forward_impl(self, model, dataset, batch, compute_cl):
            poi_weight = model.poi_embedding.weight[:-1]
            geo_gate_pois_embs = torch.multiply(
                poi_weight,
                torch.sigmoid(torch.matmul(poi_weight, model.w_gate_geo) + model.b_gate_geo),
            )
            seq_gate_pois_embs = torch.multiply(
                poi_weight,
                torch.sigmoid(torch.matmul(poi_weight, model.w_gate_seq) + model.b_gate_seq),
            )
            col_gate_pois_embs = torch.multiply(
                poi_weight,
                torch.sigmoid(torch.matmul(poi_weight, model.w_gate_col) + model.b_gate_col),
            )

            hg_pois_embs = model.mv_hconv_network(col_gate_pois_embs, dataset.pad_all_train_sessions, dataset.HG_up, dataset.HG_pu)
            hg_users = torch.sparse.mm(dataset.HG_up, hg_pois_embs)
            hg_batch_users = hg_users[batch["user_idx"]]

            geo_pois_embs = model.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)
            geo_users = torch.sparse.mm(dataset.HG_up, geo_pois_embs)
            geo_batch_users = geo_users[batch["user_idx"]]

            trans_pois_embs = model.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)
            trans_users = torch.sparse.mm(dataset.HG_up, trans_pois_embs)
            trans_batch_users = trans_users[batch["user_idx"]]

            if compute_cl:
                loss_cl_poi = model.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
                loss_cl_user = model.cal_loss_cl_users(hg_batch_users, geo_batch_users, trans_batch_users)
            else:
                zero = poi_weight.sum() * 0.0
                loss_cl_poi = zero
                loss_cl_user = zero

            norm_hg_pois = F.normalize(hg_pois_embs, p=2, dim=1)
            norm_geo_pois = F.normalize(geo_pois_embs, p=2, dim=1)
            norm_trans_pois = F.normalize(trans_pois_embs, p=2, dim=1)
            norm_hg_users = F.normalize(hg_batch_users, p=2, dim=1)
            norm_geo_users = F.normalize(geo_batch_users, p=2, dim=1)
            norm_trans_users = F.normalize(trans_batch_users, p=2, dim=1)

            hyper_coef = model.hyper_gate(norm_hg_users)
            geo_coef = model.gcn_gate(norm_geo_users)
            trans_coef = model.trans_gate(norm_trans_users)
            fused_users = hyper_coef * norm_hg_users + geo_coef * norm_geo_users + trans_coef * norm_trans_users
            fused_pois = norm_hg_pois + norm_geo_pois + norm_trans_pois
            prediction = fused_users @ fused_pois.T
            return prediction, loss_cl_user, loss_cl_poi

        def forward(self, dataset, batch, compute_cl=True):
            base_scores, loss_cl_user, loss_cl_poi = self._base_forward(dataset, batch, compute_cl=compute_cl)
            poi_emb = F.normalize(self.dchl.poi_embedding.weight[:-1], p=2, dim=1)
            query = self.rel_embedding(batch["rel_idx"])
            if bool(self.args.use_time_embedding):
                query = query + float(self.args.time_score_weight) * self.time_embedding(batch["time_bin"])
            query = F.normalize(self.query_layer_norm(query), p=2, dim=1)
            query_scores = query @ poi_emb.T
            scores = base_scores + float(self.args.query_score_weight) * query_scores
            return scores, loss_cl_user, loss_cl_poi

    return _THGDCHLModel


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_bs{args.batch_size}"
        f"_d{args.emb_dim}_mv{args.num_mv_layers}_geo{args.num_geo_layers}"
        f"_di{args.num_di_layers}_lr{args.lr:g}_cl{args.lambda_cl:g}"
    )
    suffix = str(args.save).strip()
    if suffix and suffix != "fair":
        name = f"{name}_{suffix}"
    return osp.join("results_dchl_fair", args.dataset, f"seed{args.seed}", name)


def make_loader(torch, event_dataset, batch_size, shuffle, args, device):
    from torch.utils.data import DataLoader

    return DataLoader(
        event_dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=lambda batch: collate_thg_events(
            torch,
            device,
            event_dataset.max_time_norm,
            int(args.num_time_bins),
            batch,
        ),
    )


def train_one_epoch(torch, nn, args, model, graph_data, loader, optimizer, scheduler, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    losses = []
    losses_rec = []
    losses_cl_user = []
    losses_cl_poi = []
    sync_device(torch, device)
    start = time.perf_counter()
    total_batches = len(loader)
    progress_every = int(args.progress_every)
    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        scores, loss_cl_user, loss_cl_poi = model(graph_data, batch, compute_cl=float(args.lambda_cl) > 0.0)
        loss_rec = criterion(scores, batch["label"])
        loss = loss_rec + float(args.lambda_cl) * (loss_cl_user + loss_cl_poi)
        loss.backward()
        if float(args.grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_norm))
        optimizer.step()
        if scheduler is not None and args.scheduler_step == "batch":
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
        losses_rec.append(float(loss_rec.detach().cpu().item()))
        losses_cl_user.append(float(loss_cl_user.detach().cpu().item()))
        losses_cl_poi.append(float(loss_cl_poi.detach().cpu().item()))
        if progress_every > 0 and (batch_idx % progress_every == 0 or batch_idx == total_batches):
            print(
                f"[DCHL-Fair] epoch={epoch} train_batch={batch_idx}/{total_batches} "
                f"loss={losses[-1]:.5f}",
                flush=True,
            )
    if scheduler is not None and args.scheduler_step == "epoch":
        scheduler.step()
    sync_device(torch, device)
    elapsed = time.perf_counter() - start
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "loss": mean(losses),
        "loss_rec": mean(losses_rec),
        "loss_cl_user": mean(losses_cl_user),
        "loss_cl_poi": mean(losses_cl_poi),
        "train_time_s": float(elapsed),
    }


def make_eval_batch(torch, device, batch_np, model_t, max_time_norm, num_time_bins):
    t = np.full(len(batch_np), int(model_t), dtype=np.int64)
    denom = max(int(max_time_norm) + 1, 1)
    time_bin = np.floor(t.astype(np.float64) * int(num_time_bins) / denom).astype(np.int64)
    time_bin = np.clip(time_bin, 0, int(num_time_bins) - 1)
    return {
        "user_idx": torch.from_numpy(batch_np[:, 0].astype(np.int64, copy=False)).long().to(device),
        "rel_idx": torch.from_numpy(batch_np[:, 1].astype(np.int64, copy=False)).long().to(device),
        "time_idx": torch.from_numpy(t).long().to(device),
        "time_bin": torch.from_numpy(time_bin).long().to(device),
        "label": torch.from_numpy(batch_np[:, 2].astype(np.int64, copy=False)).long().to(device),
    }


def evaluate_split(
    torch,
    args,
    model,
    graph_data,
    split_name,
    snapshot_list,
    raw_times,
    data,
    device,
    measure_forward=False,
):
    model.eval()
    sums = {}
    forward_time = 0.0
    sample_count = 0
    neg_sampler = data["negative_sampler"]
    business_first = int(data["business_first_id"])
    num_pois = int(data["num_businesses"])
    max_time_norm = int(data["timestamps_norm_max"])
    total_samples = int(sum(len(events) for events, _, _ in snapshot_list))
    total_snapshots = int(len(snapshot_list))
    progress_every = int(args.progress_every)
    eval_batches = 0
    if progress_every > 0:
        print(
            f"[DCHL-Fair] eval_start split={split_name} snapshots={total_snapshots} "
            f"samples={total_samples} measure_forward={bool(measure_forward)}",
            flush=True,
        )
    with torch.no_grad():
        for snap_idx, (events, t_norm, _) in enumerate(snapshot_list):
            if len(events) == 0:
                continue
            raw_t = int(raw_times[snap_idx])
            for batch, neg_arr, neg_mask in collect_eval_batch(
                events, raw_t, neg_sampler, split_name, int(args.eval_batch_size)
            ):
                if len(batch) == 0:
                    continue
                eval_batches += 1
                eval_batch = batch.copy()
                eval_batch[:, 2] = eval_batch[:, 2] - business_first
                batch_dict = make_eval_batch(
                    torch,
                    device,
                    eval_batch,
                    int(t_norm),
                    max_time_norm,
                    int(args.num_time_bins),
                )
                if measure_forward:
                    sync_device(torch, device)
                    t0 = time.perf_counter()
                scores, _, _ = model(graph_data, batch_dict, compute_cl=False)
                if measure_forward:
                    sync_device(torch, device)
                    forward_time += time.perf_counter() - t0

                pos_idx = torch.from_numpy((batch[:, 2] - business_first).astype(np.int64, copy=False)).long().to(device)
                pos_scores = scores.gather(1, pos_idx.view(-1, 1)).detach().cpu().numpy().astype(np.float32)

                neg_clean = neg_arr.copy()
                neg_clean[neg_clean < 0] = business_first
                neg_local = neg_clean.astype(np.int64, copy=False) - business_first
                neg_local = np.clip(neg_local, 0, num_pois - 1)
                neg_idx = torch.from_numpy(neg_local).long().to(device)
                neg_scores = scores.gather(1, neg_idx).detach().cpu().numpy().astype(np.float32)
                batch_sums = compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask)
                add_metric_sums(sums, batch_sums)
                sample_count += int(len(batch))
                if progress_every > 0 and (eval_batches % progress_every == 0 or sample_count == total_samples):
                    print(
                        f"[DCHL-Fair] eval_progress split={split_name} "
                        f"batches={eval_batches} samples={sample_count}/{total_samples}",
                        flush=True,
                    )
    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def make_scheduler(torch, args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.n_epochs)))
    raise ValueError(f"unsupported scheduler: {args.lr_scheduler}")


def run(args):
    torch, nn, F, sp, DCHL = import_dchl()
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if bool(args.enable_cudnn):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    else:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    if not data.get("is_thg", False):
        raise ValueError("train_dchl_fair.py is for Yelp-* THG datasets only.")
    describe_loaded_data(data, prefix="[DCHL-Fair]")

    train_events4 = snapshots_to_events4(data["train_list"])
    val_raw_times = snapshot_raw_times(data["val_list"])
    test_raw_times = snapshot_raw_times(data["test_list"])
    print("[DCHL-Fair] building train-only DCHL graphs...", flush=True)
    phase_start = time.perf_counter()
    graph_data = THGGraphForDCHL(torch, sp, data, train_events4, args, device)
    print(f"[DCHL-Fair] graph build done in {time.perf_counter() - phase_start:.2f}s", flush=True)

    print("[DCHL-Fair] building THG event dataset...", flush=True)
    phase_start = time.perf_counter()
    event_dataset = THGEventDataset(
        train_events4,
        data["business_first_id"],
        data["num_businesses"],
        data["timestamps_norm_max"],
        args.num_time_bins,
        device,
    )
    print(f"[DCHL-Fair] event dataset done in {time.perf_counter() - phase_start:.2f}s", flush=True)
    train_loader = make_loader(torch, event_dataset, args.batch_size, True, args, device)

    ModelClass = make_thg_dchl_model_class(torch, nn, F, DCHL)
    model = ModelClass(
        data["num_users"],
        data["num_businesses"],
        data["num_rels"],
        data["timestamps_norm_max"],
        args,
        device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.decay))
    scheduler = make_scheduler(torch, args, optimizer)

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, vars(args))
    checkpoint_path = osp.join(out_dir, "best_model.pt")

    print(
        f"[DCHL-Fair] model users={data['num_users']} businesses={data['num_businesses']} "
        f"rels={data['num_rels']} device={device} train_events={len(train_events4)}",
        flush=True,
    )
    print(
        f"[DCHL-Fair] graph nnz: HG_pu={graph_data.HG_pu._nnz()} "
        f"geo={graph_data.poi_geo_graph._nnz()} trans={graph_data.HG_poi_src._nnz()}",
        flush=True,
    )

    best_val = -float("inf")
    best_epoch = 0
    train_time_total = 0.0
    epoch_logs = []
    early_stopped = False
    early_stop_epoch = 0
    reset_cuda_peaks(torch, device)
    for epoch in range(1, int(args.n_epochs) + 1):
        log = train_one_epoch(torch, nn, args, model, graph_data, train_loader, optimizer, scheduler, device, epoch)
        log["epoch"] = int(epoch)
        train_time_total += float(log["train_time_s"])

        do_val = (epoch % int(args.evaluate_every)) == 0
        if do_val:
            val_metrics, _ = evaluate_split(
                torch,
                args,
                model,
                graph_data,
                "val",
                data["val_list"],
                val_raw_times,
                data,
                device,
                measure_forward=False,
            )
            log["val_mrr_strict"] = float(val_metrics["mrr_strict"])
            log["val_hit@1_strict"] = float(val_metrics["hit@1_strict"])
            log["val_hit@10_strict"] = float(val_metrics["hit@10_strict"])
            if val_metrics["mrr_strict"] > best_val + float(args.tolerance):
                best_val = float(val_metrics["mrr_strict"])
                best_epoch = int(epoch)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, checkpoint_path)
            log["epochs_since_best"] = int(epoch - best_epoch) if best_epoch else 0

        epoch_logs.append(log)
        print(
            f"[DCHL-Fair] epoch={epoch} loss={log['loss']:.5f} rec={log['loss_rec']:.5f} "
            f"cl_user={log['loss_cl_user']:.5f} cl_poi={log['loss_cl_poi']:.5f} "
            f"train_time={log['train_time_s']:.2f}s best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if do_val and best_epoch and int(epoch - best_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[DCHL-Fair] early stop at epoch={epoch}: no val_mrr improvement for "
                f"{epoch - best_epoch} epochs (patience={args.patience})",
                flush=True,
            )
            break

    train_peak_allocated = cuda_peak_allocated(torch, device)
    train_peak_reserved = cuda_peak_reserved(torch, device)

    if osp.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        best_epoch = int(epoch_logs[-1]["epoch"]) if epoch_logs else 0
        torch.save({"state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": {}}, checkpoint_path)

    val_metrics, _ = evaluate_split(
        torch,
        args,
        model,
        graph_data,
        "val",
        data["val_list"],
        val_raw_times,
        data,
        device,
        measure_forward=False,
    )
    reset_cuda_peaks(torch, device)
    test_metrics, test_profile = evaluate_split(
        torch,
        args,
        model,
        graph_data,
        "test",
        data["test_list"],
        test_raw_times,
        data,
        device,
        measure_forward=True,
    )
    eval_peak_allocated = cuda_peak_allocated(torch, device)
    eval_peak_reserved = cuda_peak_reserved(torch, device)

    metrics = {
        "format": "dchl_fair_thg_v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "ns_q": int(args.ns_q),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "best_epoch": int(best_epoch),
        "best_val_mrr": float(best_val),
        "early_stopped": bool(early_stopped),
        "early_stop_epoch": int(early_stop_epoch),
        "patience": int(args.patience),
        "train_time_s": float(train_time_total),
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "eval_peak_allocated_bytes": eval_peak_allocated,
        "eval_peak_reserved_bytes": eval_peak_reserved,
        "test_forward_time_s": float(test_profile["forward_time_s"]),
        "test_inference_sample_count": int(test_profile["sample_count"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_mrr": float(val_metrics["mrr_strict"]),
        "val_hit1": float(val_metrics["hit@1_strict"]),
        "val_hit10": float(val_metrics["hit@10_strict"]),
        "test_mrr": float(test_metrics["mrr_strict"]),
        "test_hit1": float(test_metrics["hit@1_strict"]),
        "test_hit10": float(test_metrics["hit@10_strict"]),
        "epoch_logs": epoch_logs,
        "model_note": (
            "Official baseline_DCHL model components are reused for collaborative hypergraph, "
            "geographical graph, directed transition hypergraph, adaptive fusion, and cross-view "
            "contrastive learning. THG adaptation maps businesses to local POI ids, injects friend "
            "edges into the user-POI hypergraph, and adds a thin relation/time query scorer. "
            "Final ranking is computed per query over exactly one positive plus the protocol "
            "business negatives returned by utils.collect_eval_batch."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[DCHL-Fair] final val_mrr={metrics['val_mrr']:.6f} "
        f"test_mrr={metrics['test_mrr']:.6f} test_hit1={metrics['test_hit1']:.6f} "
        f"test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[DCHL-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak_alloc={format_bytes(train_peak_allocated)} "
        f"train_peak_reserved={format_bytes(train_peak_reserved)} "
        f"eval_peak_alloc={format_bytes(eval_peak_allocated)} "
        f"eval_peak_reserved={format_bytes(eval_peak_reserved)}",
        flush=True,
    )
    print(f"[DCHL-Fair] saved -> {out_dir}", flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair DCHL trainer for Yelp ST-THG business prediction.")
    parser.add_argument("--dataset", type=str, default="Yelp-BOI", choices=list(eagle_utils.THG_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--evaluate_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=9999)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=("none", "step", "cosine"))
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--scheduler_step", type=str, default="epoch", choices=("epoch", "batch"))
    parser.add_argument("--save", type=str, default="fair")
    parser.add_argument("--progress_every", type=int, default=100)

    parser.add_argument("--distance_threshold", default=2.5, type=float)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda_cl", type=float, default=0.1)
    parser.add_argument("--num_mv_layers", type=int, default=3)
    parser.add_argument("--num_geo_layers", type=int, default=3)
    parser.add_argument("--num_di_layers", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--keep_rate", type=float, default=1.0)
    parser.add_argument("--keep_rate_poi", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.1)

    parser.add_argument("--query_score_weight", type=float, default=1.0)
    parser.add_argument("--use_time_embedding", action="store_true", default=True)
    parser.add_argument("--no_time_embedding", dest="use_time_embedding", action="store_false")
    parser.add_argument("--time_score_weight", type=float, default=0.1)
    parser.add_argument("--num_time_bins", type=int, default=128)
    parser.add_argument("--use_friend_edges", action="store_true", default=True)
    parser.add_argument("--no_friend_edges", dest="use_friend_edges", action="store_false")
    parser.add_argument("--friend_weight", type=float, default=0.2)
    parser.add_argument("--friend_undirected", action="store_true", default=True)
    parser.add_argument("--directed_friend_edges", dest="friend_undirected", action="store_false")
    parser.add_argument("--transition_window", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--geo_chunk_size", type=int, default=512)
    parser.add_argument("--enable_cudnn", action="store_true", default=False)
    parser.add_argument("--cudnn_benchmark", action="store_true", default=False)

    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if not 0.0 <= float(args.train_predict_ratio) <= 1.0:
        raise ValueError("--train_predict_ratio must be in [0, 1]")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("--batch_size and --eval_batch_size must be positive")
    if int(args.n_epochs) <= 0:
        raise ValueError("--n_epochs must be positive")
    if int(args.evaluate_every) <= 0:
        raise ValueError("--evaluate_every must be positive")
    if int(args.patience) <= 0:
        raise ValueError("--patience must be positive")
    if int(args.emb_dim) <= 0:
        raise ValueError("--emb_dim must be positive")
    if int(args.num_time_bins) <= 0:
        raise ValueError("--num_time_bins must be positive")
    if int(args.progress_every) < 0:
        raise ValueError("--progress_every must be >= 0")
    if float(args.distance_threshold) <= 0:
        raise ValueError("--distance_threshold must be positive")
    if not 0.0 < float(args.keep_rate) <= 1.0:
        raise ValueError("--keep_rate must be in (0, 1]")
    if not 0.0 < float(args.keep_rate_poi) <= 1.0:
        raise ValueError("--keep_rate_poi must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
