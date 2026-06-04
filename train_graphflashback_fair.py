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
GRAPHFB_DIR = osp.join(REPO_DIR, "baseline_GraphFlashback")


def import_graphflashback():
    if not osp.isdir(GRAPHFB_DIR):
        raise FileNotFoundError(f"GraphFlashback directory not found: {GRAPHFB_DIR}")
    saved_utils = sys.modules.pop("utils", None)
    if GRAPHFB_DIR in sys.path:
        sys.path.remove(GRAPHFB_DIR)
    sys.path.insert(0, GRAPHFB_DIR)
    try:
        model_path = osp.join(GRAPHFB_DIR, "network.py")
        spec = importlib.util.spec_from_file_location("baseline_graphflashback_network", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import GraphFlashback network from {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved_utils is not None:
            sys.modules["utils"] = saved_utils
    import torch
    import torch.nn as nn
    import scipy.sparse as sp

    return torch, nn, sp, module.Flashback, module.RnnFactory


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


def time_slot_from_raw(raw_t):
    hour = (int(raw_t) // 3600) % 24
    day = (int(raw_t) // 86400) % 7
    return int(day * 24 + hour)


def append_snapshot_to_histories(histories, events, raw_t, business_first, business_lat, business_lon, args):
    slot = time_slot_from_raw(raw_t)
    for s, _, o in events.astype(np.int64, copy=False):
        local = int(o) - int(business_first)
        if 0 <= local < len(business_lat):
            histories[int(s)].append((local, float(raw_t), slot, float(business_lat[local]), float(business_lon[local])))


def fill_history(seq, times, slots, coords, history, pad_id, args):
    maxlen = int(args.sequence_length)
    if not history:
        seq[:] = pad_id
        times[:] = 0.0
        slots[:] = 0
        coords[:] = 0.0
        return
    tail = history[-maxlen:]
    start = maxlen - len(tail)
    seq[:] = pad_id
    times[:] = 0.0
    slots[:] = 0
    coords[:] = 0.0
    for i, (item, raw_t, slot, lat, lon) in enumerate(tail, start=start):
        seq[i] = int(item)
        times[i] = float(raw_t)
        slots[i] = int(slot)
        coords[i, 0] = float(lat)
        coords[i, 1] = float(lon)


def business_geo_arrays(data):
    itemnum = int(data["num_businesses"])
    business_first = int(data["business_first_id"])
    lat = np.zeros(itemnum, dtype=np.float32)
    lon = np.zeros(itemnum, dtype=np.float32)
    ids = np.asarray(data["business_ids"], dtype=np.int64)
    local = ids - business_first
    valid = (local >= 0) & (local < itemnum)
    lat[local[valid]] = np.asarray(data["business_latitude"], dtype=np.float32)[valid]
    lon[local[valid]] = np.asarray(data["business_longitude"], dtype=np.float32)[valid]
    return lat, lon


def haversine_edges(lat, lon, threshold_km):
    itemnum = len(lat)
    earth_km = 6371.0
    lat_rad = np.radians(lat.astype(np.float64))
    lon_rad = np.radians(lon.astype(np.float64))
    rows = []
    cols = []
    vals = []
    chunk = 512
    for start in range(0, itemnum, chunk):
        end = min(itemnum, start + chunk)
        dlat = lat_rad[start:end, None] - lat_rad[None, :]
        dlon = lon_rad[start:end, None] - lon_rad[None, :]
        a = np.sin(dlat * 0.5) ** 2 + np.cos(lat_rad[start:end, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon * 0.5) ** 2
        dist = 2.0 * earth_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        rr, cc = np.where((dist <= float(threshold_km)) & (dist > 0.0))
        if len(rr):
            rows.append(rr.astype(np.int64) + start)
            cols.append(cc.astype(np.int64))
            vals.append(np.ones(len(rr), dtype=np.float32))
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def random_walk_matrix(sp, matrix):
    matrix = sp.coo_matrix(matrix)
    rowsum = np.asarray(matrix.sum(1)).reshape(-1)
    inv = np.zeros_like(rowsum, dtype=np.float32)
    nz = rowsum > 0
    inv[nz] = 1.0 / rowsum[nz]
    return sp.diags(inv).dot(matrix).tocoo()


def scipy_to_torch_sparse(torch, matrix):
    coo = matrix.tocoo()
    idx = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    return torch.sparse_coo_tensor(
        torch.from_numpy(idx).long(),
        torch.from_numpy(coo.data.astype(np.float32, copy=False)),
        coo.shape,
        dtype=torch.float32,
    ).coalesce()


def build_graphs(sp, data, args, business_lat, business_lon):
    itemnum = int(data["num_businesses"])
    loc_count = itemnum + 1
    pad_id = itemnum
    business_first = int(data["business_first_id"])
    user_count = int(data["num_users"])

    user_items = [[] for _ in range(user_count)]
    rows = []
    cols = []
    vals = []
    for events, _, _ in data["train_list"]:
        for s, _, o in events.astype(np.int64, copy=False):
            local = int(o) - business_first
            if 0 <= local < itemnum:
                user_items[int(s)].append(local)
    for seq in user_items:
        for a, b in zip(seq[:-1], seq[1:]):
            rows.append(a)
            cols.append(b)
            vals.append(1.0)
            if bool(args.symmetrize_transition):
                rows.append(b)
                cols.append(a)
                vals.append(1.0)

    if bool(args.merge_spatial_into_transition):
        sr, sc, sv = haversine_edges(business_lat, business_lon, args.spatial_threshold_km)
        if len(sr):
            rows.extend(sr.tolist())
            cols.extend(sc.tolist())
            vals.extend((sv * float(args.spatial_merge_weight)).tolist())

    transition = sp.coo_matrix((vals, (rows, cols)), shape=(loc_count, loc_count), dtype=np.float32)

    if bool(args.use_spatial_graph):
        sr, sc, sv = haversine_edges(business_lat, business_lon, args.spatial_threshold_km)
        spatial_graph = sp.coo_matrix((sv, (sr, sc)), shape=(loc_count, loc_count), dtype=np.float32)
    else:
        spatial_graph = None

    inter_rows = []
    inter_cols = []
    inter_vals = []
    for u, seq in enumerate(user_items):
        seen = set()
        for item in seq:
            if item not in seen:
                inter_rows.append(u)
                inter_cols.append(item)
                inter_vals.append(1.0)
                seen.add(item)

    if bool(args.use_friend_interact) and float(args.friend_interact_weight) > 0.0:
        friend_edges = data.get("static_user_friend_edges")
        if friend_edges is not None and len(friend_edges):
            topk = max(1, int(args.friend_interact_topk))
            recent = []
            for seq in user_items:
                seen = []
                for item in reversed(seq):
                    if item not in seen:
                        seen.append(item)
                    if len(seen) == topk:
                        break
                recent.append(seen)
            for u, v in friend_edges.astype(np.int64, copy=False):
                if 0 <= int(u) < user_count and 0 <= int(v) < user_count:
                    for item in recent[int(v)]:
                        inter_rows.append(int(u))
                        inter_cols.append(int(item))
                        inter_vals.append(float(args.friend_interact_weight))
                    if bool(args.friend_undirected):
                        for item in recent[int(u)]:
                            inter_rows.append(int(v))
                            inter_cols.append(int(item))
                            inter_vals.append(float(args.friend_interact_weight))

    interact = sp.csr_matrix((inter_vals, (inter_rows, inter_cols)), shape=(user_count, loc_count), dtype=np.float32)
    friend_graph = None
    if bool(args.use_graph_user):
        f_rows = []
        f_cols = []
        f_vals = []
        friend_edges = data.get("static_user_friend_edges")
        if friend_edges is not None and len(friend_edges):
            for u, v in friend_edges.astype(np.int64, copy=False):
                if 0 <= int(u) < user_count and 0 <= int(v) < user_count:
                    f_rows.append(int(u))
                    f_cols.append(int(v))
                    f_vals.append(1.0)
                    if bool(args.friend_undirected):
                        f_rows.append(int(v))
                        f_cols.append(int(u))
                        f_vals.append(1.0)
        f_rows.extend(range(user_count))
        f_cols.extend(range(user_count))
        f_vals.extend([1.0] * user_count)
        friend_graph = random_walk_matrix(
            sp,
            sp.coo_matrix((f_vals, (f_rows, f_cols)), shape=(user_count, user_count), dtype=np.float32),
        )
    return transition, spatial_graph, friend_graph, interact


class THGFlashbackSamples:
    def __init__(self, users, rels, t_norms, labels, x, times, slots, coords, args, torch, device):
        self.users = users.astype(np.int64, copy=False)
        self.rels = rels.astype(np.int64, copy=False)
        self.t_norms = t_norms.astype(np.int64, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.x = x.astype(np.int64, copy=False)
        self.times = times.astype(np.float32, copy=False)
        self.slots = slots.astype(np.int64, copy=False)
        self.coords = coords.astype(np.float32, copy=False)
        self.args = args
        self.torch = torch
        self.device = device

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, idx):
        idx = int(idx)
        return (
            self.users[idx],
            self.rels[idx],
            self.t_norms[idx],
            self.labels[idx],
            self.x[idx],
            self.times[idx],
            self.slots[idx],
            self.coords[idx],
        )

    def collate(self, batch):
        users, rels, t_norms, labels, x, times, slots, coords = zip(*batch)
        torch = self.torch
        x = np.stack(x).astype(np.int64, copy=False).T
        times = np.stack(times).astype(np.float32, copy=False).T
        slots = np.stack(slots).astype(np.int64, copy=False).T
        coords = np.stack(coords).astype(np.float32, copy=False).transpose(1, 0, 2)
        batch_size = len(labels)
        return {
            "active_users": torch.from_numpy(np.asarray(users, dtype=np.int64)).long().to(self.device),
            "rels": torch.from_numpy(np.asarray(rels, dtype=np.int64)).long().to(self.device),
            "t_norms": torch.from_numpy(np.asarray(t_norms, dtype=np.int64)).long().to(self.device),
            "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64)).long().to(self.device),
            "x": torch.from_numpy(x).long().to(self.device),
            "t": torch.from_numpy(times).float().to(self.device),
            "t_slot": torch.from_numpy(slots).long().to(self.device),
            "s": torch.from_numpy(coords).float().to(self.device),
            "y_t": torch.zeros(int(self.args.sequence_length), batch_size, dtype=torch.float32, device=self.device),
            "y_t_slot": torch.zeros(int(self.args.sequence_length), batch_size, dtype=torch.long, device=self.device),
            "y_s": torch.zeros(int(self.args.sequence_length), batch_size, 2, dtype=torch.float32, device=self.device),
        }


def build_train_arrays(data, args, business_lat, business_lon):
    total = sum(len(events) for events, _, _ in data["train_list"])
    seq_len = int(args.sequence_length)
    itemnum = int(data["num_businesses"])
    business_first = int(data["business_first_id"])
    histories = [[] for _ in range(int(data["num_users"]))]
    users = np.zeros(total, dtype=np.int64)
    rels = np.zeros(total, dtype=np.int64)
    t_norms = np.zeros(total, dtype=np.int64)
    labels = np.zeros(total, dtype=np.int64)
    x = np.full((total, seq_len), itemnum, dtype=np.int64)
    times = np.zeros((total, seq_len), dtype=np.float32)
    slots = np.zeros((total, seq_len), dtype=np.int64)
    coords = np.zeros((total, seq_len, 2), dtype=np.float32)
    row = 0
    for events, t_norm, raw_t in data["train_list"]:
        for s, r, o in events.astype(np.int64, copy=False):
            users[row] = int(s)
            rels[row] = int(r)
            t_norms[row] = int(t_norm)
            labels[row] = int(o) - business_first
            fill_history(x[row], times[row], slots[row], coords[row], histories[int(s)], itemnum, args)
            row += 1
        append_snapshot_to_histories(histories, events, raw_t, business_first, business_lat, business_lon, args)
    return users, rels, t_norms, labels, x, times, slots, coords


def init_histories(data, seed_snapshot_lists, args, business_lat, business_lon):
    histories = [[] for _ in range(int(data["num_users"]))]
    business_first = int(data["business_first_id"])
    for snapshot_list in seed_snapshot_lists:
        for events, _, raw_t in snapshot_list:
            append_snapshot_to_histories(histories, events, raw_t, business_first, business_lat, business_lon, args)
    return histories


def make_eval_batch(torch, device, args, data, batch_events, t_norm, histories, business_lat, business_lon):
    seq_len = int(args.sequence_length)
    itemnum = int(data["num_businesses"])
    batch_size = len(batch_events)
    x = np.full((batch_size, seq_len), itemnum, dtype=np.int64)
    times = np.zeros((batch_size, seq_len), dtype=np.float32)
    slots = np.zeros((batch_size, seq_len), dtype=np.int64)
    coords = np.zeros((batch_size, seq_len, 2), dtype=np.float32)
    for i, (s, _, _) in enumerate(batch_events.astype(np.int64, copy=False)):
        fill_history(x[i], times[i], slots[i], coords[i], histories[int(s)], itemnum, args)
    return {
        "active_users": torch.from_numpy(batch_events[:, 0].astype(np.int64, copy=False)).long().to(device),
        "rels": torch.from_numpy(batch_events[:, 1].astype(np.int64, copy=False)).long().to(device),
        "t_norms": torch.full((batch_size,), int(t_norm), dtype=torch.long, device=device),
        "x": torch.from_numpy(x.T).long().to(device),
        "t": torch.from_numpy(times.T).float().to(device),
        "t_slot": torch.from_numpy(slots.T).long().to(device),
        "s": torch.from_numpy(coords.transpose(1, 0, 2)).float().to(device),
        "y_t": torch.zeros(seq_len, batch_size, dtype=torch.float32, device=device),
        "y_t_slot": torch.zeros(seq_len, batch_size, dtype=torch.long, device=device),
        "y_s": torch.zeros(seq_len, batch_size, 2, dtype=torch.float32, device=device),
    }


def make_model_class(torch, nn, Flashback, RnnFactory):
    class THGGraphFlashback(nn.Module):
        def __init__(self, loc_count, user_count, num_rels, args, transition_graph, spatial_graph, friend_graph, interact_graph):
            super().__init__()
            def f_t(delta_t, user_len):
                day = float(args.time_period_seconds)
                return ((torch.cos(delta_t * 2 * np.pi / day) + 1.0) / 2.0) * torch.exp(-(delta_t / day * float(args.lambda_t)))

            def f_s(delta_s, user_len):
                return torch.exp(-(delta_s * float(args.lambda_s)))

            self.args = args
            self.loc_count = int(loc_count)
            self.itemnum = int(loc_count) - 1
            self.flashback = Flashback(
                loc_count,
                user_count,
                args.hidden_dim,
                f_t,
                f_s,
                RnnFactory(args.rnn),
                args.lambda_loc,
                args.lambda_user,
                args.use_weight,
                transition_graph,
                spatial_graph,
                friend_graph,
                args.use_graph_user,
                args.use_spatial_graph,
                interact_graph,
            )
            if bool(args.use_graph_user) and friend_graph is not None:
                self.flashback.friend_graph = scipy_to_torch_sparse(torch, friend_graph)
                if bool(args.use_weight):
                    self.flashback.user_gconv_weight = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
            self.rel_embedding = nn.Embedding(num_rels, args.hidden_dim)
            self.time_embedding = nn.Embedding(args.num_time_bins, args.hidden_dim)
            self.rel_item_bias = nn.Embedding(num_rels, loc_count)
            nn.init.xavier_uniform_(self.rel_embedding.weight)
            nn.init.xavier_uniform_(self.time_embedding.weight)
            nn.init.zeros_(self.rel_item_bias.weight)

        def initial_h(self, batch_size, device):
            if self.args.rnn == "lstm":
                h = torch.zeros(1, batch_size, self.args.hidden_dim, device=device)
                c = torch.zeros(1, batch_size, self.args.hidden_dim, device=device)
                return h, c
            return torch.zeros(1, batch_size, self.args.hidden_dim, device=device)

        def score(self, batch):
            batch_size = int(batch["active_users"].shape[0])
            with torch.no_grad():
                self.flashback.encoder.weight.data[self.itemnum].zero_()
            h = self.initial_h(batch_size, batch["active_users"].device)
            out, _ = self.flashback(
                batch["x"],
                batch["t"],
                batch["t_slot"],
                batch["s"],
                batch["y_t"],
                batch["y_t_slot"],
                batch["y_s"],
                h,
                batch["active_users"],
            )
            scores = out[-1]
            if float(self.args.rel_score_weight) != 0.0:
                rel_vec = self.rel_embedding(batch["rels"])
                loc_vec = self.flashback.encoder.weight
                scores = scores + float(self.args.rel_score_weight) * rel_vec.matmul(loc_vec.transpose(0, 1))
            if bool(self.args.use_time_embedding) and float(self.args.time_score_weight) != 0.0:
                time_bins = torch.clamp(batch["t_norms"] * int(self.args.num_time_bins) // max(1, int(self.args.max_time_norm) + 1), 0, int(self.args.num_time_bins) - 1)
                time_vec = self.time_embedding(time_bins)
                loc_vec = self.flashback.encoder.weight
                scores = scores + float(self.args.time_score_weight) * time_vec.matmul(loc_vec.transpose(0, 1))
            if bool(self.args.use_rel_item_bias):
                scores = scores + float(self.args.rel_bias_weight) * self.rel_item_bias(batch["rels"])
            scores = scores.clone()
            scores[:, self.itemnum] = -1e16
            return scores

    return THGGraphFlashback


def make_loader(torch, dataset, batch_size, shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, collate_fn=dataset.collate)


def make_scheduler(torch, args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.n_epochs)))
    raise ValueError(f"unsupported lr scheduler: {args.lr_scheduler}")


def train_one_epoch(torch, nn, args, model, loader, optimizer, scheduler, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    losses = []
    sync_device(torch, device)
    start = time.perf_counter()
    total_batches = len(loader)
    progress_every = int(args.progress_every)
    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        scores = model.score(batch)
        loss = criterion(scores, batch["labels"])
        loss.backward()
        if float(args.grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_norm))
        optimizer.step()
        if scheduler is not None and args.scheduler_step == "batch":
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
        if progress_every > 0 and (batch_idx % progress_every == 0 or batch_idx == total_batches):
            print(
                f"[GraphFlashback-Fair] epoch={epoch} train_batch={batch_idx}/{total_batches} "
                f"loss={losses[-1]:.5f}",
                flush=True,
            )
    if scheduler is not None and args.scheduler_step == "epoch":
        scheduler.step()
    sync_device(torch, device)
    return {"loss": float(np.mean(losses)) if losses else 0.0, "train_time_s": float(time.perf_counter() - start)}


def evaluate_split(torch, args, model, split_name, snapshot_list, seed_lists, data, business_lat, business_lon, device, measure_forward=False):
    model.eval()
    sums = {}
    forward_time = 0.0
    sample_count = 0
    neg_sampler = data["negative_sampler"]
    business_first = int(data["business_first_id"])
    histories = init_histories(data, seed_lists, args, business_lat, business_lon)
    total_samples = int(sum(len(events) for events, _, _ in snapshot_list))
    total_snapshots = int(len(snapshot_list))
    progress_every = int(args.progress_every)
    eval_batches = 0
    if progress_every > 0:
        print(
            f"[GraphFlashback-Fair] eval_start split={split_name} snapshots={total_snapshots} "
            f"samples={total_samples} measure_forward={bool(measure_forward)}",
            flush=True,
        )
    with torch.no_grad():
        for events, t_norm, raw_t in snapshot_list:
            if len(events) == 0:
                continue
            for batch, neg_arr, neg_mask in collect_eval_batch(events, int(raw_t), neg_sampler, split_name, int(args.eval_batch_size)):
                if len(batch) == 0:
                    continue
                eval_batches += 1
                batch_dict = make_eval_batch(torch, device, args, data, batch, t_norm, histories, business_lat, business_lon)
                if measure_forward:
                    sync_device(torch, device)
                    t0 = time.perf_counter()
                scores = model.score(batch_dict)
                if measure_forward:
                    sync_device(torch, device)
                    forward_time += time.perf_counter() - t0
                pos_idx = torch.from_numpy((batch[:, 2] - business_first).astype(np.int64, copy=False)).long().to(device)
                pos_scores = scores.gather(1, pos_idx.view(-1, 1)).detach().cpu().numpy().astype(np.float32)
                neg_clean = neg_arr.copy()
                neg_clean[neg_clean < 0] = business_first
                neg_idx_np = np.clip(neg_clean.astype(np.int64, copy=False) - business_first, 0, int(data["num_businesses"]) - 1)
                neg_idx = torch.from_numpy(neg_idx_np).long().to(device)
                neg_scores = scores.gather(1, neg_idx).detach().cpu().numpy().astype(np.float32)
                add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
                sample_count += int(len(batch))
                if progress_every > 0 and (eval_batches % progress_every == 0 or sample_count == total_samples):
                    print(
                        f"[GraphFlashback-Fair] eval_progress split={split_name} "
                        f"batches={eval_batches} samples={sample_count}/{total_samples}",
                        flush=True,
                    )
            append_snapshot_to_histories(histories, events, raw_t, business_first, business_lat, business_lon, args)
    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_bs{args.batch_size}"
        f"_d{args.hidden_dim}_L{args.sequence_length}_{args.rnn}"
        f"_lt{args.lambda_t:g}_ls{args.lambda_s:g}_lr{args.lr:g}"
    )
    suffix = str(args.save).strip()
    if suffix and suffix != "fair":
        name = f"{name}_{suffix}"
    return osp.join("results_graphflashback_fair", args.dataset, f"seed{args.seed}", name)


def run(args):
    torch, nn, sp, Flashback, RnnFactory = import_graphflashback()
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

    data = load_datasets(args.dataset, q=args.ns_q, load_train_ratio=args.train_predict_ratio, load_eval_neg=True, ns_seed=args.ns_seed)
    if not data.get("is_thg", False):
        raise ValueError("train_graphflashback_fair.py is for Yelp-* THG datasets only.")
    describe_loaded_data(data, prefix="[GraphFlashback-Fair]")

    print("[GraphFlashback-Fair] building business geo arrays and train-only graphs...", flush=True)
    phase_start = time.perf_counter()
    business_lat, business_lon = business_geo_arrays(data)
    transition_graph, spatial_graph, friend_graph, interact_graph = build_graphs(sp, data, args, business_lat, business_lon)
    print(f"[GraphFlashback-Fair] graph build done in {time.perf_counter() - phase_start:.2f}s", flush=True)
    print("[GraphFlashback-Fair] building Flashback train sequences...", flush=True)
    phase_start = time.perf_counter()
    users, rels, t_norms, labels, x, times, slots, coords = build_train_arrays(data, args, business_lat, business_lon)
    print(f"[GraphFlashback-Fair] train sequence build done in {time.perf_counter() - phase_start:.2f}s", flush=True)
    train_dataset = THGFlashbackSamples(users, rels, t_norms, labels, x, times, slots, coords, args, torch, device)
    train_loader = make_loader(torch, train_dataset, args.batch_size, True)

    loc_count = int(data["num_businesses"]) + 1
    args.max_time_norm = int(data["timestamps_norm_max"])
    ModelClass = make_model_class(torch, nn, Flashback, RnnFactory)
    model = ModelClass(loc_count, data["num_users"], data["num_rels"], args, transition_graph, spatial_graph, friend_graph, interact_graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = make_scheduler(torch, args, optimizer)

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, vars(args))
    checkpoint_path = osp.join(out_dir, "best_model.pt")

    print(
        f"[GraphFlashback-Fair] model users={data['num_users']} businesses={data['num_businesses']} "
        f"rels={data['num_rels']} loc_count={loc_count} train_samples={len(train_dataset)} device={device}",
        flush=True,
    )
    print(
        f"[GraphFlashback-Fair] graph nnz transition={transition_graph.nnz} "
        f"spatial={0 if spatial_graph is None else spatial_graph.nnz} interact={interact_graph.nnz}",
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
        log = train_one_epoch(torch, nn, args, model, train_loader, optimizer, scheduler, device, epoch)
        log["epoch"] = int(epoch)
        train_time_total += float(log["train_time_s"])
        do_val = (epoch % int(args.evaluate_every)) == 0
        if do_val:
            val_metrics, _ = evaluate_split(
                torch, args, model, "val", data["val_list"], [data["train_list"]],
                data, business_lat, business_lon, device, measure_forward=False
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
            f"[GraphFlashback-Fair] epoch={epoch} loss={log['loss']:.5f} "
            f"train_time={log['train_time_s']:.2f}s best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if do_val and best_epoch and int(epoch - best_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[GraphFlashback-Fair] early stop at epoch={epoch}: no val_mrr improvement for "
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
        torch, args, model, "val", data["val_list"], [data["train_list"]],
        data, business_lat, business_lon, device, measure_forward=False
    )
    reset_cuda_peaks(torch, device)
    test_metrics, test_profile = evaluate_split(
        torch, args, model, "test", data["test_list"], [data["train_list"], data["val_list"]],
        data, business_lat, business_lon, device, measure_forward=True
    )
    eval_peak_allocated = cuda_peak_allocated(torch, device)
    eval_peak_reserved = cuda_peak_reserved(torch, device)

    metrics = {
        "format": "graphflashback_fair_thg_v1",
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
            "Official baseline_GraphFlashback Flashback/RNN implementation is reused. Businesses are "
            "mapped to local POI ids and one extra padding POI is added for short THG histories. "
            "Train and eval samples use only the user's history before the current query timestamp. "
            "Transition, spatial, and user-POI interaction graphs are built from train-only data; "
            "friend edges optionally augment user-POI preference. Final metrics are computed per query "
            "over one positive plus the protocol business negatives from utils.collect_eval_batch."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[GraphFlashback-Fair] final val_mrr={metrics['val_mrr']:.6f} "
        f"test_mrr={metrics['test_mrr']:.6f} test_hit1={metrics['test_hit1']:.6f} "
        f"test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[GraphFlashback-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak_alloc={format_bytes(train_peak_allocated)} "
        f"train_peak_reserved={format_bytes(train_peak_reserved)} "
        f"eval_peak_alloc={format_bytes(eval_peak_allocated)} "
        f"eval_peak_reserved={format_bytes(eval_peak_reserved)}",
        flush=True,
    )
    print(f"[GraphFlashback-Fair] saved -> {out_dir}", flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair GraphFlashback trainer for Yelp ST-THG business prediction.")
    parser.add_argument("--dataset", type=str, default="Yelp-BOI", choices=list(eagle_utils.THG_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_norm", type=float, default=5.0)
    parser.add_argument("--evaluate_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=9999)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="step", choices=("none", "step", "cosine"))
    parser.add_argument("--lr_step_size", type=int, default=20)
    parser.add_argument("--lr_gamma", type=float, default=0.2)
    parser.add_argument("--scheduler_step", type=str, default="epoch", choices=("epoch", "batch"))
    parser.add_argument("--save", type=str, default="fair")
    parser.add_argument("--progress_every", type=int, default=100)

    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--rnn", type=str, default="gru", choices=("rnn", "gru", "lstm"))
    parser.add_argument("--lambda_t", type=float, default=0.1)
    parser.add_argument("--lambda_s", type=float, default=100.0)
    parser.add_argument("--lambda_loc", type=float, default=1.0)
    parser.add_argument("--lambda_user", type=float, default=1.0)
    parser.add_argument("--time_period_seconds", type=float, default=86400.0)
    parser.add_argument("--use_weight", action="store_true", default=False)
    parser.add_argument("--use_graph_user", action="store_true", default=False)
    parser.add_argument("--use_spatial_graph", action="store_true", default=False)
    parser.add_argument("--merge_spatial_into_transition", action="store_true", default=True)
    parser.add_argument("--no_merge_spatial_into_transition", dest="merge_spatial_into_transition", action="store_false")
    parser.add_argument("--spatial_threshold_km", type=float, default=2.5)
    parser.add_argument("--spatial_merge_weight", type=float, default=0.2)
    parser.add_argument("--symmetrize_transition", action="store_true", default=True)
    parser.add_argument("--directed_transition", dest="symmetrize_transition", action="store_false")
    parser.add_argument("--use_friend_interact", action="store_true", default=True)
    parser.add_argument("--no_friend_interact", dest="use_friend_interact", action="store_false")
    parser.add_argument("--friend_interact_weight", type=float, default=0.1)
    parser.add_argument("--friend_interact_topk", type=int, default=3)
    parser.add_argument("--friend_undirected", action="store_true", default=True)
    parser.add_argument("--directed_friend", dest="friend_undirected", action="store_false")
    parser.add_argument("--rel_score_weight", type=float, default=0.5)
    parser.add_argument("--use_time_embedding", action="store_true", default=True)
    parser.add_argument("--no_time_embedding", dest="use_time_embedding", action="store_false")
    parser.add_argument("--time_score_weight", type=float, default=0.1)
    parser.add_argument("--num_time_bins", type=int, default=128)
    parser.add_argument("--use_rel_item_bias", action="store_true", default=True)
    parser.add_argument("--no_rel_item_bias", dest="use_rel_item_bias", action="store_false")
    parser.add_argument("--rel_bias_weight", type=float, default=0.5)
    parser.add_argument("--enable_cudnn", action="store_true", default=False)
    parser.add_argument("--cudnn_benchmark", action="store_true", default=False)

    args = parser.parse_args()
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if not 0.0 <= float(args.train_predict_ratio) <= 1.0:
        raise ValueError("--train_predict_ratio must be in [0, 1]")
    if int(args.batch_size) <= 0 or int(args.eval_batch_size) <= 0:
        raise ValueError("--batch_size and --eval_batch_size must be positive")
    if int(args.n_epochs) <= 0 or int(args.evaluate_every) <= 0 or int(args.patience) <= 0:
        raise ValueError("--n_epochs, --evaluate_every, and --patience must be positive")
    if int(args.sequence_length) <= 0 or int(args.hidden_dim) <= 0 or int(args.num_time_bins) <= 0:
        raise ValueError("--sequence_length, --hidden_dim, and --num_time_bins must be positive")
    if int(args.progress_every) < 0:
        raise ValueError("--progress_every must be >= 0")
    if int(args.friend_interact_topk) <= 0:
        raise ValueError("--friend_interact_topk must be positive")
    if float(args.spatial_threshold_km) <= 0:
        raise ValueError("--spatial_threshold_km must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
