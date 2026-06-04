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
AGRAN_DIR = osp.join(REPO_DIR, "baseline_AGRAN")


def ensure_agran_import_path():
    if not osp.isdir(AGRAN_DIR):
        raise FileNotFoundError(f"AGRAN directory not found: {AGRAN_DIR}")
    if AGRAN_DIR in sys.path:
        sys.path.remove(AGRAN_DIR)
    sys.path.insert(0, AGRAN_DIR)


def import_agran():
    ensure_agran_import_path()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    model_path = osp.join(AGRAN_DIR, "model_ag.py")
    spec = importlib.util.spec_from_file_location("baseline_agran_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import AGRAN model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, nn, F, module.AGRAN


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


def time_value(raw_t, t_norm, args):
    bucket_seconds = int(args.time_bucket_seconds)
    if bucket_seconds > 0:
        return int(raw_t) // bucket_seconds + 1
    return int(t_norm) + 1


def append_snapshot_to_histories(histories, events, t_norm, raw_t, business_first, args):
    tv = time_value(raw_t, t_norm, args)
    for s, _, o in events.astype(np.int64, copy=False):
        item = int(o) - int(business_first) + 1
        if item > 0:
            histories[int(s)].append((item, tv))


def fill_prefix(seq_row, time_row, history, maxlen):
    if not history:
        return
    tail = history[-int(maxlen) :]
    start = int(maxlen) - len(tail)
    seq_row[start:] = [x[0] for x in tail]
    time_row[start:] = [x[1] for x in tail]


def build_business_geo_arrays(data):
    itemnum = int(data["num_businesses"])
    business_first = int(data["business_first_id"])
    lat = np.zeros(itemnum + 1, dtype=np.float64)
    lon = np.zeros(itemnum + 1, dtype=np.float64)
    ids = np.asarray(data["business_ids"], dtype=np.int64)
    lats = np.asarray(data["business_latitude"], dtype=np.float64)
    lons = np.asarray(data["business_longitude"], dtype=np.float64)
    local = ids - business_first + 1
    valid = (local >= 1) & (local <= itemnum)
    lat[local[valid]] = lats[valid]
    lon[local[valid]] = lons[valid]
    return lat, lon


def haversine_matrix_km(lat, lon):
    earth_km = 6371.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = lat_rad[:, :, None] - lat_rad[:, None, :]
    dlon = lon_rad[:, :, None] - lon_rad[:, None, :]
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat_rad[:, :, None]) * np.cos(lat_rad[:, None, :]) * np.sin(dlon * 0.5) ** 2
    return 2.0 * earth_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def make_time_matrices(time_seqs, time_span):
    diff = np.abs(time_seqs[:, :, None].astype(np.int64) - time_seqs[:, None, :].astype(np.int64))
    return np.minimum(diff, int(time_span)).astype(np.int64, copy=False)


def make_distance_matrices(seqs, lat_by_item, lon_by_item, dis_span):
    lat = lat_by_item[seqs]
    lon = lon_by_item[seqs]
    dist = np.floor(haversine_matrix_km(lat, lon)).astype(np.int64, copy=False)
    dist = np.minimum(dist, int(dis_span))
    pad = (seqs == 0)
    dist[pad[:, :, None] | pad[:, None, :]] = int(dis_span)
    return dist.astype(np.int64, copy=False)


def time_bins_from_norm(t_norms, max_time_norm, num_time_bins):
    denom = max(int(max_time_norm) + 1, 1)
    bins = np.floor(np.asarray(t_norms, dtype=np.float64) * int(num_time_bins) / denom).astype(np.int64)
    return np.clip(bins, 0, int(num_time_bins) - 1)


class AGRANTHGSamples:
    def __init__(self, users, rels, t_norms, labels, seqs, time_seqs, max_time_norm, lat_by_item, lon_by_item, args, torch, device):
        self.users = users.astype(np.int64, copy=False)
        self.rels = rels.astype(np.int64, copy=False)
        self.t_norms = t_norms.astype(np.int64, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.seqs = seqs.astype(np.int64, copy=False)
        self.time_seqs = time_seqs.astype(np.int64, copy=False)
        self.max_time_norm = int(max_time_norm)
        self.lat_by_item = lat_by_item
        self.lon_by_item = lon_by_item
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
            self.seqs[idx],
            self.time_seqs[idx],
        )

    def collate(self, batch):
        users, rels, t_norms, labels, seqs, time_seqs = zip(*batch)
        users = np.asarray(users, dtype=np.int64)
        rels = np.asarray(rels, dtype=np.int64)
        t_norms = np.asarray(t_norms, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        seqs = np.stack(seqs).astype(np.int64, copy=False)
        time_seqs = np.stack(time_seqs).astype(np.int64, copy=False)
        time_matrices = make_time_matrices(time_seqs, self.args.time_span)
        dis_matrices = make_distance_matrices(seqs, self.lat_by_item, self.lon_by_item, self.args.dis_span)
        torch = self.torch
        return {
            "users_np": users,
            "seqs_np": seqs,
            "time_matrices_np": time_matrices,
            "dis_matrices_np": dis_matrices,
            "rels": torch.from_numpy(rels).long().to(self.device),
            "time_bins": torch.from_numpy(time_bins_from_norm(t_norms, self.max_time_norm, self.args.num_time_bins)).long().to(self.device),
            "labels": torch.from_numpy(labels).long().to(self.device),
        }


def build_train_arrays(data, args):
    train_list = data["train_list"]
    total = sum(len(events) for events, _, _ in train_list)
    maxlen = int(args.maxlen)
    users = np.zeros(total, dtype=np.int64)
    rels = np.zeros(total, dtype=np.int64)
    t_norms = np.zeros(total, dtype=np.int64)
    labels = np.zeros(total, dtype=np.int64)
    seqs = np.zeros((total, maxlen), dtype=np.int32)
    time_seqs = np.zeros((total, maxlen), dtype=np.int32)
    histories = [[] for _ in range(int(data["num_users"]))]
    business_first = int(data["business_first_id"])
    row = 0
    for events, t_norm, raw_t in train_list:
        tv = time_value(raw_t, t_norm, args)
        for s, r, o in events.astype(np.int64, copy=False):
            users[row] = int(s)
            rels[row] = int(r)
            t_norms[row] = int(t_norm)
            labels[row] = int(o) - business_first + 1
            fill_prefix(seqs[row], time_seqs[row], histories[int(s)], maxlen)
            row += 1
        append_snapshot_to_histories(histories, events, t_norm, raw_t, business_first, args)
    return users, rels, t_norms, labels, seqs, time_seqs


def init_histories_from_snapshots(data, snapshot_lists, args):
    histories = [[] for _ in range(int(data["num_users"]))]
    business_first = int(data["business_first_id"])
    for snapshot_list in snapshot_lists:
        for events, t_norm, raw_t in snapshot_list:
            append_snapshot_to_histories(histories, events, t_norm, raw_t, business_first, args)
    return histories


def make_eval_batch_dict(torch, device, args, data, batch_events, t_norm, raw_t, histories, lat_by_item, lon_by_item):
    maxlen = int(args.maxlen)
    batch_size = int(len(batch_events))
    seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
    time_seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
    for i, (s, _, _) in enumerate(batch_events.astype(np.int64, copy=False)):
        fill_prefix(seqs[i], time_seqs[i], histories[int(s)], maxlen)
    return {
        "users_np": batch_events[:, 0].astype(np.int64, copy=False),
        "seqs_np": seqs,
        "time_matrices_np": make_time_matrices(time_seqs, args.time_span),
        "dis_matrices_np": make_distance_matrices(seqs, lat_by_item, lon_by_item, args.dis_span),
        "rels": torch.from_numpy(batch_events[:, 1].astype(np.int64, copy=False)).long().to(device),
        "time_bins": torch.from_numpy(time_bins_from_norm(np.full(batch_size, int(t_norm), dtype=np.int64), data["timestamps_norm_max"], args.num_time_bins)).long().to(device),
    }


def mask_for_kl(torch, adj, epsilon=0.0, mask_value=-1e16):
    mask = (adj > float(epsilon)).detach().float()
    return adj * mask + (1.0 - mask) * float(mask_value)


def build_transition_prior(data, args):
    itemnum = int(data["num_businesses"])
    business_first = int(data["business_first_id"])
    user_items = [[] for _ in range(int(data["num_users"]))]
    rows = []
    cols = []
    vals = []
    for events, _, _ in data["train_list"]:
        for s, _, o in events.astype(np.int64, copy=False):
            user_items[int(s)].append(int(o) - business_first)
    for seq in user_items:
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if 0 <= a < itemnum and 0 <= b < itemnum:
                rows.append(a)
                cols.append(b)
                vals.append(1.0)
                if bool(args.symmetrize_prior):
                    rows.append(b)
                    cols.append(a)
                    vals.append(1.0)
    if bool(args.prior_self_loop):
        rows.extend(range(itemnum))
        cols.extend(range(itemnum))
        vals.extend([float(args.self_loop_prior_weight)] * itemnum)

    if bool(args.use_friend_prior) and float(args.friend_prior_weight) > 0.0:
        friend_edges = data.get("static_user_friend_edges")
        if friend_edges is not None and len(friend_edges):
            topk = max(1, int(args.friend_prior_topk))
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
                if not (0 <= int(u) < len(recent) and 0 <= int(v) < len(recent)):
                    continue
                left = recent[int(u)]
                right = recent[int(v)]
                if not left or not right:
                    continue
                for a in left:
                    for b in right:
                        rows.append(int(a))
                        cols.append(int(b))
                        vals.append(float(args.friend_prior_weight))
                        if bool(args.friend_undirected_prior):
                            rows.append(int(b))
                            cols.append(int(a))
                            vals.append(float(args.friend_prior_weight))

    prior = np.zeros((itemnum, itemnum), dtype=np.float32)
    if rows:
        np.add.at(prior, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)), np.asarray(vals, dtype=np.float32))
    return prior


def make_agran_thg_model_class(torch, nn, F, AGRAN):
    class AGRANTHGModel(nn.Module):
        def __init__(self, usernum, itemnum, num_rels, args):
            super().__init__()
            self.args = args
            self.itemnum = int(itemnum)
            self.agran = AGRAN(usernum, itemnum, itemnum, args)
            self.rel_embedding = nn.Embedding(num_rels, args.hidden_units)
            self.time_embedding = nn.Embedding(args.num_time_bins, args.hidden_units)
            self.query_norm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.rel_item_bias = nn.Embedding(num_rels, itemnum + 1)
            nn.init.xavier_uniform_(self.rel_embedding.weight)
            nn.init.xavier_uniform_(self.time_embedding.weight)
            nn.init.zeros_(self.rel_item_bias.weight)

        def score(self, batch):
            item_embs, support = self.agran.gcn(self.agran.item_emb)
            log_feats = self.agran.seq2feats(
                batch["users_np"],
                batch["seqs_np"],
                batch["time_matrices_np"],
                batch["dis_matrices_np"],
                item_embs,
            )
            final_feat = log_feats[:, -1, :]
            query = final_feat
            if float(self.args.rel_score_weight) != 0.0:
                query = query + float(self.args.rel_score_weight) * self.rel_embedding(batch["rels"])
            if bool(self.args.use_time_embedding) and float(self.args.time_score_weight) != 0.0:
                query = query + float(self.args.time_score_weight) * self.time_embedding(batch["time_bins"])
            query = self.query_norm(query)
            scores = query.matmul(item_embs.transpose(0, 1))
            if bool(self.args.use_rel_item_bias):
                scores = scores + float(self.args.rel_bias_weight) * self.rel_item_bias(batch["rels"])
            scores = scores.clone()
            scores[:, 0] = -1e16
            return scores, support

    return AGRANTHGModel


def make_loader(torch, dataset, batch_size, shuffle):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=dataset.collate,
    )


def make_scheduler(torch, args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_size), gamma=float(args.lr_gamma))
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.n_epochs)))
    raise ValueError(f"unsupported scheduler: {args.lr_scheduler}")


def train_one_epoch(torch, nn, args, model, loader, optimizer, scheduler, prior_prob, device, epoch):
    model.train()
    ce = nn.CrossEntropyLoss().to(device)
    kl_loss = nn.KLDivLoss(reduction="batchmean").to(device)
    losses = []
    losses_ce = []
    losses_kl = []
    sync_device(torch, device)
    start = time.perf_counter()
    total_batches = len(loader)
    progress_every = int(args.progress_every)
    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        scores, support = model.score(batch)
        loss_ce = ce(scores, batch["labels"])
        if float(args.kl_reg) > 0.0:
            support_log = torch.log(torch.softmax(mask_for_kl(torch, support), dim=-1) + 1e-9)
            loss_kl = kl_loss(support_log, prior_prob)
        else:
            loss_kl = scores.sum() * 0.0
        loss = loss_ce + float(args.kl_reg) * loss_kl
        loss.backward()
        if float(args.grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_norm))
        optimizer.step()
        if scheduler is not None and args.scheduler_step == "batch":
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
        losses_ce.append(float(loss_ce.detach().cpu().item()))
        losses_kl.append(float(loss_kl.detach().cpu().item()))
        if progress_every > 0 and (batch_idx % progress_every == 0 or batch_idx == total_batches):
            print(
                f"[AGRAN-Fair] epoch={epoch} train_batch={batch_idx}/{total_batches} "
                f"loss={losses[-1]:.5f} ce={losses_ce[-1]:.5f} kl={losses_kl[-1]:.5f}",
                flush=True,
            )
    if scheduler is not None and args.scheduler_step == "epoch":
        scheduler.step()
    sync_device(torch, device)
    elapsed = time.perf_counter() - start
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "loss": mean(losses),
        "loss_ce": mean(losses_ce),
        "loss_kl": mean(losses_kl),
        "train_time_s": float(elapsed),
    }


def evaluate_split(torch, args, model, split_name, snapshot_list, history_seed_lists, data, lat_by_item, lon_by_item, device, measure_forward=False):
    model.eval()
    sums = {}
    forward_time = 0.0
    sample_count = 0
    neg_sampler = data["negative_sampler"]
    business_first = int(data["business_first_id"])
    histories = init_histories_from_snapshots(data, history_seed_lists, args)
    total_samples = int(sum(len(events) for events, _, _ in snapshot_list))
    total_snapshots = int(len(snapshot_list))
    progress_every = int(args.progress_every)
    eval_batches = 0
    if progress_every > 0:
        print(
            f"[AGRAN-Fair] eval_start split={split_name} snapshots={total_snapshots} "
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
                batch_dict = make_eval_batch_dict(torch, device, args, data, batch, t_norm, raw_t, histories, lat_by_item, lon_by_item)
                if measure_forward:
                    sync_device(torch, device)
                    t0 = time.perf_counter()
                scores, _ = model.score(batch_dict)
                if measure_forward:
                    sync_device(torch, device)
                    forward_time += time.perf_counter() - t0
                pos_idx = torch.from_numpy((batch[:, 2] - business_first + 1).astype(np.int64, copy=False)).long().to(device)
                pos_scores = scores.gather(1, pos_idx.view(-1, 1)).detach().cpu().numpy().astype(np.float32)

                neg_clean = neg_arr.copy()
                neg_clean[neg_clean < 0] = business_first
                neg_idx_np = (neg_clean.astype(np.int64, copy=False) - business_first + 1)
                neg_idx_np = np.clip(neg_idx_np, 1, int(data["num_businesses"]))
                neg_idx = torch.from_numpy(neg_idx_np).long().to(device)
                neg_scores = scores.gather(1, neg_idx).detach().cpu().numpy().astype(np.float32)
                add_metric_sums(sums, compute_ranking_metric_sums(pos_scores, neg_scores, neg_mask))
                sample_count += int(len(batch))
                if progress_every > 0 and (eval_batches % progress_every == 0 or sample_count == total_samples):
                    print(
                        f"[AGRAN-Fair] eval_progress split={split_name} "
                        f"batches={eval_batches} samples={sample_count}/{total_samples}",
                        flush=True,
                    )
            append_snapshot_to_histories(histories, events, t_norm, raw_t, business_first, args)
    metrics = finalize_metric_sums(sums)
    metrics["mrr"] = metrics["mrr_strict"]
    metrics["hit1"] = metrics["hit@1_strict"]
    metrics["hit10"] = metrics["hit@10_strict"]
    return metrics, {"forward_time_s": float(forward_time), "sample_count": int(sample_count)}


def make_out_dir(args):
    name = (
        f"nsq{args.ns_q}_ns{args.ns_seed}_bs{args.batch_size}"
        f"_d{args.hidden_units}_L{args.maxlen}_blk{args.num_blocks}"
        f"_h{args.num_heads}_lr{args.lr:g}_kl{args.kl_reg:g}"
    )
    suffix = str(args.save).strip()
    if suffix and suffix != "fair":
        name = f"{name}_{suffix}"
    return osp.join("results_agran_fair", args.dataset, f"seed{args.seed}", name)


def run(args):
    torch, nn, F, AGRAN = import_agran()
    set_random_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.device = str(device)
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
        raise ValueError("train_agran_fair.py is for Yelp-* THG datasets only.")
    describe_loaded_data(data, prefix="[AGRAN-Fair]")

    print("[AGRAN-Fair] building business geo arrays and train sequences...", flush=True)
    phase_start = time.perf_counter()
    lat_by_item, lon_by_item = build_business_geo_arrays(data)
    users, rels, t_norms, labels, seqs, time_seqs = build_train_arrays(data, args)
    print(f"[AGRAN-Fair] train sequence build done in {time.perf_counter() - phase_start:.2f}s", flush=True)
    print("[AGRAN-Fair] building AGRAN train dataset and transition prior...", flush=True)
    phase_start = time.perf_counter()
    train_dataset = AGRANTHGSamples(
        users,
        rels,
        t_norms,
        labels,
        seqs,
        time_seqs,
        data["timestamps_norm_max"],
        lat_by_item,
        lon_by_item,
        args,
        torch,
        device,
    )
    train_loader = make_loader(torch, train_dataset, args.batch_size, True)

    prior_np = build_transition_prior(data, args)
    prior = torch.from_numpy(prior_np).float().to(device)
    with torch.no_grad():
        prior_prob = torch.softmax(mask_for_kl(torch, prior), dim=-1)
    del prior
    print(f"[AGRAN-Fair] dataset/prior build done in {time.perf_counter() - phase_start:.2f}s", flush=True)

    ModelClass = make_agran_thg_model_class(torch, nn, F, AGRAN)
    model = ModelClass(data["num_users"], data["num_businesses"], data["num_rels"], args).to(device)
    for name, param in model.named_parameters():
        if name == "rel_item_bias.weight":
            torch.nn.init.zeros_(param.data)
            continue
        if param.dim() > 1:
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except Exception:
                pass
    with torch.no_grad():
        model.agran.item_emb.weight.data[0].zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.9, 0.98), weight_decay=float(args.l2_emb))
    scheduler = make_scheduler(torch, args, optimizer)

    out_dir = make_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, vars(args))
    checkpoint_path = osp.join(out_dir, "best_model.pt")

    print(
        f"[AGRAN-Fair] model users={data['num_users']} businesses={data['num_businesses']} "
        f"rels={data['num_rels']} train_samples={len(train_dataset)} device={device}",
        flush=True,
    )
    print(
        f"[AGRAN-Fair] prior nnz={int(np.count_nonzero(prior_np))} "
        f"maxlen={args.maxlen} time_span={args.time_span} dis_span={args.dis_span}",
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
        log = train_one_epoch(torch, nn, args, model, train_loader, optimizer, scheduler, prior_prob, device, epoch)
        log["epoch"] = int(epoch)
        train_time_total += float(log["train_time_s"])

        do_val = (epoch % int(args.evaluate_every)) == 0
        if do_val:
            val_metrics, _ = evaluate_split(
                torch,
                args,
                model,
                "val",
                data["val_list"],
                [data["train_list"]],
                data,
                lat_by_item,
                lon_by_item,
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
            f"[AGRAN-Fair] epoch={epoch} loss={log['loss']:.5f} ce={log['loss_ce']:.5f} "
            f"kl={log['loss_kl']:.5f} train_time={log['train_time_s']:.2f}s "
            f"best_val_mrr={max(best_val, 0.0):.5f}",
            flush=True,
        )
        if do_val and best_epoch and int(epoch - best_epoch) >= int(args.patience):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[AGRAN-Fair] early stop at epoch={epoch}: no val_mrr improvement for "
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
        "val",
        data["val_list"],
        [data["train_list"]],
        data,
        lat_by_item,
        lon_by_item,
        device,
        measure_forward=False,
    )
    reset_cuda_peaks(torch, device)
    test_metrics, test_profile = evaluate_split(
        torch,
        args,
        model,
        "test",
        data["test_list"],
        [data["train_list"], data["val_list"]],
        data,
        lat_by_item,
        lon_by_item,
        device,
        measure_forward=True,
    )
    eval_peak_allocated = cuda_peak_allocated(torch, device)
    eval_peak_reserved = cuda_peak_reserved(torch, device)

    metrics = {
        "format": "agran_fair_thg_v1",
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
            "Official baseline_AGRAN AGRAN/AGCN modules are reused. Businesses are mapped to "
            "AGRAN item ids 1..num_businesses with 0 as padding. Each THG query uses the user's "
            "history before the current timestamp to build AGRAN sequence, time-relation, and "
            "geographical distance matrices. AGCN is regularized by a train-only transition prior, "
            "optionally augmented with static friend co-preference edges. Relation/time embeddings "
            "lightly condition the final query representation. Final metrics are computed per query "
            "over one positive plus the protocol business negatives from utils.collect_eval_batch."
        ),
    }
    save_metrics(out_dir, metrics)
    print(
        f"[AGRAN-Fair] final val_mrr={metrics['val_mrr']:.6f} "
        f"test_mrr={metrics['test_mrr']:.6f} test_hit1={metrics['test_hit1']:.6f} "
        f"test_hit10={metrics['test_hit10']:.6f}",
        flush=True,
    )
    print(
        f"[AGRAN-Fair] train_time={train_time_total:.3f}s "
        f"test_forward_time={test_profile['forward_time_s']:.3f}s "
        f"test_samples={test_profile['sample_count']} "
        f"train_peak_alloc={format_bytes(train_peak_allocated)} "
        f"train_peak_reserved={format_bytes(train_peak_reserved)} "
        f"eval_peak_alloc={format_bytes(eval_peak_allocated)} "
        f"eval_peak_reserved={format_bytes(eval_peak_reserved)}",
        flush=True,
    )
    print(f"[AGRAN-Fair] saved -> {out_dir}", flush=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser("Fair AGRAN trainer for Yelp ST-THG business prediction.")
    parser.add_argument("--dataset", type=str, default="Yelp-BOI", choices=list(eagle_utils.THG_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=121)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_emb", type=float, default=1e-4)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=9999)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=("none", "step", "cosine"))
    parser.add_argument("--lr_step_size", type=int, default=40)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--scheduler_step", type=str, default="epoch", choices=("epoch", "batch"))
    parser.add_argument("--save", type=str, default="fair")
    parser.add_argument("--progress_every", type=int, default=100)

    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--time_span", type=int, default=256)
    parser.add_argument("--dis_span", type=int, default=256)
    parser.add_argument("--kl_reg", type=float, default=1.0)

    parser.add_argument("--time_bucket_seconds", type=int, default=86400)
    parser.add_argument("--num_time_bins", type=int, default=256)
    parser.add_argument("--rel_score_weight", type=float, default=1.0)
    parser.add_argument("--use_time_embedding", action="store_true", default=True)
    parser.add_argument("--no_time_embedding", dest="use_time_embedding", action="store_false")
    parser.add_argument("--time_score_weight", type=float, default=0.2)
    parser.add_argument("--use_rel_item_bias", action="store_true", default=True)
    parser.add_argument("--no_rel_item_bias", dest="use_rel_item_bias", action="store_false")
    parser.add_argument("--rel_bias_weight", type=float, default=1.0)
    parser.add_argument("--symmetrize_prior", action="store_true", default=True)
    parser.add_argument("--directed_prior", dest="symmetrize_prior", action="store_false")
    parser.add_argument("--prior_self_loop", action="store_true", default=True)
    parser.add_argument("--no_prior_self_loop", dest="prior_self_loop", action="store_false")
    parser.add_argument("--self_loop_prior_weight", type=float, default=0.1)
    parser.add_argument("--use_friend_prior", action="store_true", default=True)
    parser.add_argument("--no_friend_prior", dest="use_friend_prior", action="store_false")
    parser.add_argument("--friend_prior_weight", type=float, default=0.05)
    parser.add_argument("--friend_prior_topk", type=int, default=2)
    parser.add_argument("--friend_undirected_prior", action="store_true", default=True)
    parser.add_argument("--directed_friend_prior", dest="friend_undirected_prior", action="store_false")
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
    if int(args.maxlen) <= 0 or int(args.hidden_units) <= 0:
        raise ValueError("--maxlen and --hidden_units must be positive")
    if int(args.num_heads) <= 0 or int(args.hidden_units) % int(args.num_heads) != 0:
        raise ValueError("--hidden_units must be divisible by --num_heads")
    if int(args.num_blocks) <= 0:
        raise ValueError("--num_blocks must be positive")
    if int(args.time_span) <= 0 or int(args.dis_span) <= 0 or int(args.num_time_bins) <= 0:
        raise ValueError("--time_span, --dis_span, and --num_time_bins must be positive")
    if int(args.progress_every) < 0:
        raise ValueError("--progress_every must be >= 0")
    if int(args.friend_prior_topk) <= 0:
        raise ValueError("--friend_prior_topk must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
