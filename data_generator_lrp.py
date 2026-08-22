# data_generator_lrp.py  —  Fixed-Dataset Edition (Nazari et al. NeurIPS 2018)
# ==============================================================================
# CHANGES vs original:
#
#   [1] SIZE_CONFIGS dict — تعریف مرکزی برای همه سایزها (lrp3, lrp8, lrp10, lrp20)
#       هر سایز n_cust و n_fac مستقل دارد.
#
#   [2] LRPDataGenerator.__init__ یک پارامتر اختیاری "size_tag" می‌گیرد که
#       اگر داده شود (مثلاً 'lrp8') مقادیر n_cust و n_fac را از SIZE_CONFIGS
#       می‌خواند؛ وگرنه رفتار قبلی (خواندن از args) حفظ می‌شود.
#       → سازگاری کامل با train_lrp.py و lrp_export.py موجود.
#
#   [3] export_instance_for_gams() (متد جدید) — یک instance با seed ثابت
#       می‌سازد و فایل‌های CSV/JSON را برای GAMS می‌نویسد.
#
#   [4] task_specific_params_for_size() — برگرداندن TaskLRP برای سایز دلخواه
#       بدون نیاز به ویرایش task_specific_params.py.
#
# استفاده سریع برای مقایسه با روش دقیق (8 مشتری):
#
#   python data_generator_lrp.py --size lrp8 --seed 42
#   python make_inc.py --tag lrp_n8_seed42
#   gams lrp_benchmark.gms
#   python lrp_drl_infer.py --tag lrp_n8_seed42 --model_path logs/.../model.ckpt
#   python lrp_compare.py
# ==============================================================================

import numpy as np
import pandas as pd
import os, json, argparse
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from collections import namedtuple

# ==============================================================================
# [1]  تعریف مرکزی سایزها
# ==============================================================================
SIZE_CONFIGS = {
    'lrp3':  {'n_cust':   3, 'n_fac':  3, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp8':  {'n_cust':   8, 'n_fac':  4, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp10': {'n_cust':  10, 'n_fac':  5, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp15': {'n_cust':  15, 'n_fac':  6, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp20': {'n_cust':  20, 'n_fac':  8, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp25': {'n_cust':  25, 'n_fac':  9, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp30': {'n_cust':  30, 'n_fac':  9, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp35': {'n_cust':  35, 'n_fac': 10, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp40': {'n_cust':  40, 'n_fac': 11, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp45': {'n_cust':  45, 'n_fac': 11, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp50': {'n_cust':  50, 'n_fac': 12, 'capacity_per_comp': 10.0, 'num_comp': 4},
    'lrp100':{'n_cust': 100, 'n_fac': 20, 'capacity_per_comp': 10.0, 'num_comp': 4},
}

# پارامترهای ثابت مدل — باید با آموزش DRL یکسان باشد
C_AV       = 1.0
C_F        = 10.0 / 3650.0   # C_j/H: amortised facility cost (revised model)
DEMAND_MAX = 9

TaskLRP = namedtuple('TaskLRP', [
    'task_name', 'input_dim', 'n_nodes', 'n_cust', 'n_fac',
    'decode_len', 'capacity_per_compartment', 'num_compartments',
    'demand_max', 'facility_opening_cost'
])


def task_specific_params_for_size(size_tag):
    """
    [4] برگرداندن TaskLRP برای سایز دلخواه.

    مثال:
        tp = task_specific_params_for_size('lrp8')
        # سپس در configs.py یا train_lrp.py استفاده شود
    """
    if size_tag not in SIZE_CONFIGS:
        raise ValueError(f"Unknown size_tag '{size_tag}'. "
                         f"Valid: {list(SIZE_CONFIGS.keys())}")
    cfg = SIZE_CONFIGS[size_tag]
    nc, nf = cfg['n_cust'], cfg['n_fac']
    return TaskLRP(
        task_name               = 'lrp',
        input_dim               = 4,
        n_nodes                 = 1 + nc + nf,
        n_cust                  = nc,
        n_fac                   = nf,
        decode_len              = nc + nf + nc // 4 + 3,
        capacity_per_compartment= cfg['capacity_per_comp'],
        num_compartments        = cfg['num_comp'],
        demand_max              = DEMAND_MAX,
        facility_opening_cost   = C_F,
    )


_CALIFORNIA_POPULATION_CENTERS = [
    (0.02, 0.70, 2.5), (0.11, 0.67, 1.5), (0.16, 0.60, 5.0),
    (0.18, 0.55, 2.0), (0.14, 0.64, 1.5), (0.22, 0.58, 1.0),
    (0.28, 0.56, 1.0), (0.44, 0.45, 2.0), (0.52, 0.38, 1.5),
    (0.64, 0.29, 2.0), (0.51, 0.25, 2.5), (0.56, 0.20, 4.0),
    (0.58, 0.27, 1.5), (0.54, 0.24, 2.0),
]
_CA_BBOX = ((32.5, -124.5), (42.0, -114.0))


class LRPDataGenerator(object):

    def __init__(self, args=None,
                 collision_csv_path='CA_AV_Collision_2019-2024.csv',
                 bounding_box=None,
                 grid_size=200,
                 risk_model='kde',
                 size_tag=None):
        """
        Parameters
        ----------
        args              : dict  پارامترهای آموزش (مثل قبل)
        collision_csv_path: str
        bounding_box      : tuple
        grid_size         : int
        risk_model        : str
        size_tag          : str | None
            اگر داده شود (مثلاً 'lrp8') مقادیر n_cust/n_fac از SIZE_CONFIGS
            override می‌شوند. اگر None باشد رفتار قدیمی حفظ می‌شود.
        """
        self.args         = args or {}
        self.bounding_box = bounding_box or _CA_BBOX

        # ── [2] تعیین n_cust و n_fac ──────────────────────────────────────────
        if size_tag is not None:
            if size_tag not in SIZE_CONFIGS:
                raise ValueError(f"Unknown size_tag '{size_tag}'. "
                                 f"Valid: {list(SIZE_CONFIGS.keys())}")
            cfg = SIZE_CONFIGS[size_tag]
            self._n_cust = cfg['n_cust']
            self._n_fac  = cfg['n_fac']
            self._cfg    = cfg
            print(f'  [Config] size_tag={size_tag} → '
                  f'n_cust={self._n_cust}, n_fac={self._n_fac}')
        else:
            # رفتار قدیمی — سازگار با train_lrp.py
            a = self.args
            self._n_cust = a.get('n_cust',     10)
            self._n_fac  = a.get('n_fac',       5)
            self._cfg    = {
                'n_cust':            self._n_cust,
                'n_fac':             self._n_fac,
                'capacity_per_comp': a.get('capacity_per_compartment', 10.0),
                'num_comp':          a.get('num_compartments', 4),
            }

        self._batch   = self.args.get('batch_size', 128)
        self._test_sz = self.args.get('test_size', 1000)

        print('  [Stage 1] Loading collision data and fitting KDE ...')
        self._load_and_fit_kde(collision_csv_path)
        self._precompute_risk_grid(grid_size, safe_risk_threshold=0.75)
        print(f'  Risk grid: {grid_size}x{grid_size}, '
              f'{int(self._risk_grid_binary.sum())} risky cells '
              f'({100*self._risk_grid_binary.mean():.2f}% of map)')

        print(f'  [Stage 2] K-means ({self._n_fac} clusters) ...')
        self._cached_facilities = self._compute_facilities(self._n_fac)

        # ── Fixed dataset storage ──────────────────────────────────────────────
        self.train_features = None
        self.train_lambda   = None
        self._epoch_order   = None
        self._epoch_cursor  = 0

        # ── Test set storage ───────────────────────────────────────────────────
        self.test_features = None
        self.test_lambda   = None
        self.count         = 0

    # =========================================================================
    # Stage 1: KDE
    # =========================================================================
    def _load_and_fit_kde(self, csv_path):
        lat_min, lon_min = self.bounding_box[0]
        lat_max, lon_max = self.bounding_box[1]
        df = pd.read_csv(csv_path)
        df = df[(df['Latitude']  >= lat_min) & (df['Latitude']  <= lat_max)
              & (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]
        df = df.dropna(subset=['Latitude', 'Longitude'])
        pts = df[['Latitude', 'Longitude']].values
        self.collision_points = self._normalize_geo(pts)
        print(f'  {len(self.collision_points)} collision points loaded.')
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.05)
        self.kde.fit(self.collision_points)
        g = np.linspace(0, 1, 200)
        lc, lo = np.meshgrid(g, g, indexing='ij')
        gpts = np.stack([lc.ravel(), lo.ravel()], 1)
        self._kde_max = float(self.kde.score_samples(gpts).max())

    def _precompute_risk_grid(self, resolution, safe_risk_threshold=0.75):
        g = np.linspace(0, 1, resolution)
        lats, lons = np.meshgrid(g, g, indexing='ij')
        pts   = np.stack([lats.ravel(), lons.ravel()], 1)
        log_d = self.kde.score_samples(pts)
        risk  = np.exp(log_d - self._kde_max).reshape(resolution, resolution)
        self._risk_grid        = risk
        self._risk_grid_binary = (risk >= safe_risk_threshold).astype(np.uint8)
        self._risk_grid_res    = resolution
        self._risk_threshold   = safe_risk_threshold

    # =========================================================================
    # Stage 2: K-means candidate stations
    # =========================================================================
    def _normalize_geo(self, pts):
        lat_min, lon_min = self.bounding_box[0]
        lat_max, lon_max = self.bounding_box[1]
        return np.stack(
            [(pts[:, 0] - lat_min) / (lat_max - lat_min),
             (pts[:, 1] - lon_min) / (lon_max - lon_min)], 1)

    def denormalize_geo(self, pts_norm):
        lat_min, lon_min = self.bounding_box[0]
        lat_max, lon_max = self.bounding_box[1]
        lat = pts_norm[:, 0] * (lat_max - lat_min) + lat_min
        lon = pts_norm[:, 1] * (lon_max - lon_min) + lon_min
        return np.stack([lat, lon], 1)

    def _compute_facilities(self, num_facilities):
        if len(self.collision_points) < num_facilities:
            return np.random.RandomState(42).rand(
                num_facilities, 2).astype(np.float32)
        km = KMeans(n_clusters=num_facilities, n_init=5, random_state=42)
        km.fit(self.collision_points)
        centres = np.clip(km.cluster_centers_, 0.0, 1.0).astype(np.float32)
        geo = self.denormalize_geo(centres)
        for i, (lat, lon) in enumerate(geo):
            print(f'    Candidate S{i+1}: ({lat:.4f}N, {abs(lon):.4f}W)')
        return centres

    def generate_facilities(self, num_facilities, **kwargs):
        if (hasattr(self, '_cached_facilities') and
                len(self._cached_facilities) == num_facilities):
            return self._cached_facilities.copy()
        return self._compute_facilities(num_facilities)

    # =========================================================================
    # Stage 3: λ(i,j) — vectorised over batch
    # =========================================================================
    def _compute_lambda_batch_vectorized(self, coords_batch, n_samples=10):
        B, N, _ = coords_batch.shape
        lam  = np.ones((B, N, N), dtype=np.float32)
        res  = self._risk_grid_res
        ts   = np.linspace(0, 1, n_samples)

        for i in range(N):
            for j in range(i + 1, N):
                pi = coords_batch[:, i, :]
                pj = coords_batch[:, j, :]
                pts  = pi[:, None, :] + ts[None, :, None] * (pj - pi)[:, None, :]
                rows = np.clip((pts[:, :, 0] * (res - 1)).astype(int), 0, res - 1)
                cols = np.clip((pts[:, :, 1] * (res - 1)).astype(int), 0, res - 1)
                is_risky = self._risk_grid_binary[rows, cols].any(axis=1)
                lam[:, i, j] = np.where(is_risky, 0.0, 1.0)
                lam[:, j, i] = lam[:, i, j]
        return lam

    def compute_lambda_matrix(self, coords, n_samples=10, **kwargs):
        lam   = self._compute_lambda_batch_vectorized(coords[None], n_samples)[0]
        risky = int((lam == 0).sum()) // 2
        total = coords.shape[0] * (coords.shape[0] - 1) // 2
        print(f'  lambda matrix: {risky}/{total} edges risky ({100*risky/total:.1f}%)')
        return lam.astype(np.int32)

    # =========================================================================
    # Internal helpers
    # =========================================================================
    def _build_features(self, customers, demands):
        B, nc, nf = customers.shape[0], self._n_cust, self._n_fac
        depot  = np.full((B, 1, 2), 0.5, dtype=np.float32)
        facs   = np.tile(self._cached_facilities[None], (B, 1, 1))
        coords = np.concatenate([depot, customers, facs], axis=1)
        dem    = np.concatenate([
            np.zeros((B, 1,  1), np.float32),
            demands,
            np.zeros((B, nf, 1), np.float32)], axis=1)
        nt     = np.concatenate([
            np.zeros((B,  1, 1), np.float32),
            np.ones( (B, nc, 1), np.float32),
            np.full( (B, nf, 1), 2.0, np.float32)], axis=1)
        feats  = np.concatenate([coords, dem, nt], axis=2).astype(np.float32)
        return feats, coords

    def _draw_customers(self, n, nc, rng):
        """Helper for sampling customer positions from CA population centres."""
        centers  = _CALIFORNIA_POPULATION_CENTERS
        weights  = np.array([c[2] for c in centers]); weights /= weights.sum()
        total    = n * nc
        oversamp = total * 8
        ci   = rng.choice(len(centers), size=oversamp, p=weights)
        lats = rng.normal([centers[c][0] for c in ci], 0.10)
        lons = rng.normal([centers[c][1] for c in ci], 0.10)
        ok   = (lats >= 0) & (lats <= 1) & (lons >= 0) & (lons <= 1)
        lats, lons = lats[ok], lons[ok]
        if len(lats) < total:
            uf   = rng.rand(total - len(lats), 2)
            lats = np.concatenate([lats, uf[:, 0]])
            lons = np.concatenate([lons, uf[:, 1]])
        customers = np.stack([lats[:total], lons[:total]], 1)\
                      .reshape(n, nc, 2).astype(np.float32)
        demands   = rng.randint(1, DEMAND_MAX + 1,
                                size=(n, nc, 1)).astype(np.float32)
        return customers, demands

    def _sample_instances(self, n, rng, compute_lambda=True,
                          n_lambda_samples=10, verbose=True, tag=''):
        nc = self._n_cust
        customers, demands = self._draw_customers(n, nc, rng)
        features, coords   = self._build_features(customers, demands)

        if compute_lambda:
            if verbose:
                print(f'  Computing lambda for {n} {tag}instances ...', end='')
            chunk, parts = 1000, []
            for s in range(0, n, chunk):
                parts.append(self._compute_lambda_batch_vectorized(
                    coords[s:s + chunk], n_samples=n_lambda_samples))
                if verbose and n > chunk:
                    print(f' {min(s+chunk,n)}/{n}', end='\r')
            lam = np.concatenate(parts, axis=0)
            if verbose:
                print(f'  Lambda done ({n} instances).')
        else:
            n_nodes = 1 + self._n_cust + self._n_fac
            lam = np.ones((n, n_nodes, n_nodes), dtype=np.float32)
        return features, lam

    # =========================================================================
    # [3]  export_instance_for_gams — متد جدید برای مقایسه با روش دقیق
    # =========================================================================
    def export_instance_for_gams(self, seed=42, out_dir='data', tag=None):
        """
        یک instance با seed ثابت می‌سازد و فایل‌های CSV/JSON را
        برای GAMS می‌نویسد. همان instance را DRL هم استفاده می‌کند
        تا مقایسه عادلانه باشد.

        خروجی‌ها:
          {out_dir}/{tag}_nodes.csv
          {out_dir}/{tag}_lambda.csv
          {out_dir}/{tag}_dist.csv
          {out_dir}/{tag}_params.json
        """
        nc, nf = self._n_cust, self._n_fac
        if tag is None:
            tag = f'lrp_n{nc}_seed{seed}'

        print(f'\n[Export] Generating instance: '
              f'n_cust={nc}, n_fac={nf}, seed={seed}')

        rng             = np.random.RandomState(seed)
        customers, demands = self._draw_customers(1, nc, rng)
        feats, coords   = self._build_features(customers, demands)
        lam             = self._compute_lambda_batch_vectorized(
                              coords, n_samples=10)[0]
        dist_mat        = cdist(coords[0], coords[0]).astype(np.float32)

        n_nodes  = feats.shape[1]
        cust_idx = list(range(1, 1 + nc))
        fac_idx  = list(range(1 + nc, n_nodes))

        os.makedirs(out_dir, exist_ok=True)

        # ── nodes.csv ─────────────────────────────────────────────────────────
        rows = []
        for i in range(n_nodes):
            rows.append({
                'node':        i,
                'x':           float(feats[0, i, 0]),
                'y':           float(feats[0, i, 1]),
                'demand':      float(feats[0, i, 2]),
                'node_type':   int(feats[0, i, 3]),
                'is_depot':    int(i == 0),
                'is_customer': int(i in cust_idx),
                'is_facility': int(i in fac_idx),
            })
        pd.DataFrame(rows).to_csv(f'{out_dir}/{tag}_nodes.csv', index=False)

        # ── lambda.csv ────────────────────────────────────────────────────────
        lam_rows = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    lam_rows.append({'i': i, 'j': j,
                                     'lambda': int(lam[i, j])})
        pd.DataFrame(lam_rows).to_csv(f'{out_dir}/{tag}_lambda.csv', index=False)

        # ── dist.csv ──────────────────────────────────────────────────────────
        dist_rows = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist_rows.append({'i': i, 'j': j,
                                      'dist': float(dist_mat[i, j])})
        pd.DataFrame(dist_rows).to_csv(f'{out_dir}/{tag}_dist.csv', index=False)

        # ── params.json ───────────────────────────────────────────────────────
        params = {
            'n_nodes':           n_nodes,
            'n_cust':            nc,
            'n_fac':             nf,
            'n_vehicles':        nc,          # کران بالا: یک وسیله برای هر مشتری
            'n_compartments':    self._cfg['num_comp'],
            'capacity_per_comp': self._cfg['capacity_per_comp'],
            'C_AV':              C_AV,
            'C_f':               C_F,
            'depot_node':        0,
            'cust_nodes':        cust_idx,
            'fac_nodes':         fac_idx,
        }
        with open(f'{out_dir}/{tag}_params.json', 'w') as f:
            json.dump(params, f, indent=2)

        # ── خلاصه ─────────────────────────────────────────────────────────────
        risky = int((lam == 0).sum()) // 2
        total = n_nodes * (n_nodes - 1) // 2
        print(f'  Nodes    : {n_nodes}  '
              f'(depot=1, customers={nc}, facilities={nf})')
        print(f'  Risky    : {risky}/{total} edges ({100*risky/total:.1f}%)')
        print(f'  Exported : {out_dir}/{tag}_*.csv/.json')
        print(f'\n  Next steps:')
        print(f'    python make_inc.py --tag {tag} --datadir {out_dir}')
        print(f'    gams lrp_benchmark.gms')
        return tag

    # =========================================================================
    # PUBLIC — Fixed Dataset (Nazari-style)
    # =========================================================================
    def generate_fixed_dataset(self, n_instances=10000, seed=None,
                               n_lambda_samples=10, verbose=True):
        rng = np.random.RandomState(seed)
        if verbose:
            print(f'\n[Fixed Dataset] Generating {n_instances:,} '
                  f'training instances ...')
        feats, lam = self._sample_instances(
            n_instances, rng,
            compute_lambda=True,
            n_lambda_samples=n_lambda_samples,
            verbose=verbose, tag='training ')
        self.train_features = feats
        self.train_lambda   = lam
        self._epoch_order   = None
        self._epoch_cursor  = 0
        if verbose:
            n_nodes = feats.shape[1]
            n_pairs = n_nodes * (n_nodes - 1) // 2
            risky   = ((lam == 0).sum(axis=(1, 2)) // 2).mean()
            print(f'\n[Fixed Dataset] Ready.')
            print(f'  Instances : {n_instances:,}')
            print(f'  Features  : {feats.nbytes/1e6:.1f} MB')
            print(f'  Lambda    : {lam.nbytes/1e6:.1f} MB')
            print(f'  Avg risky : {risky:.1f}/{n_pairs} '
                  f'({100*risky/n_pairs:.1f}%) per instance')

    def start_epoch(self, shuffle=True, rng=None):
        if self.train_features is None:
            raise RuntimeError("Call generate_fixed_dataset() first.")
        n = len(self.train_features)
        r = rng if rng is not None else np.random
        self._epoch_order  = r.permutation(n) if shuffle else np.arange(n)
        self._epoch_cursor = 0

    def get_epoch_batches(self, batch_size, shuffle=True):
        self.start_epoch(shuffle=shuffle)
        n = len(self.train_features)
        while self._epoch_cursor + batch_size <= n:
            idx = self._epoch_order[
                  self._epoch_cursor: self._epoch_cursor + batch_size]
            yield (self.train_features[idx],
                   self.train_lambda[idx])
            self._epoch_cursor += batch_size

    @property
    def steps_per_epoch(self):
        if self.train_features is None: return 0
        return len(self.train_features) // self._batch

    @property
    def n_train_instances(self):
        return 0 if self.train_features is None else len(self.train_features)

    # =========================================================================
    # PUBLIC — Random mini-batch (online fallback)
    # =========================================================================
    def get_train_next(self):
        if self.train_features is not None:
            idx = np.random.choice(
                len(self.train_features), size=self._batch,
                replace=len(self.train_features) < self._batch)
            return self.train_features[idx], self.train_lambda[idx]
        return self._online_batch()

    def _online_batch(self):
        rng = np.random.RandomState()
        feats, lam = self._sample_instances(
            self._batch, rng, compute_lambda=True, verbose=False)
        return feats, lam

    # =========================================================================
    # PUBLIC — Test set
    # =========================================================================
    def get_test_all(self):
        if self.test_features is None:
            self._generate_test_set()
        return self.test_features, self.test_lambda

    def get_test_next(self):
        if self.test_features is None:
            self._generate_test_set()
        i = self.count % len(self.test_features)
        self.count += 1
        return self.test_features[i:i+1], self.test_lambda[i:i+1]

    def _generate_test_set(self):
        print(f'  Building test set ({self._test_sz} instances) ...')
        rng = np.random.RandomState(0)
        feats, lam = self._sample_instances(
            self._test_sz, rng, compute_lambda=True,
            n_lambda_samples=10, verbose=True, tag='test ')
        self.test_features = feats
        self.test_lambda   = lam
        print(f'  Test set: {self.test_features.shape}')

    def reset(self):
        self.count = 0

    @property
    def n_problems(self):
        return self._test_sz

    # =========================================================================
    # PUBLIC — Single instance (for visualisation / debug)
    # =========================================================================
    def generate_instance(self, num_customers, num_facilities,
                          seed=None, compute_lambda=True, **kwargs):
        rng = np.random.RandomState(seed)
        feats, lam = self._sample_instances(
            1, rng, compute_lambda=compute_lambda, verbose=True)
        return {
            'coords':    feats[0, :, :2],
            'features':  feats[0],
            'demand':    feats[0, :, 2:3],
            'node_type': feats[0, :, 3:4],
            'lambda':    lam[0].astype(np.int32),
        }

    def get_risk_map(self, resolution=100):
        g = np.linspace(0, 1, resolution)
        lr, lo = np.meshgrid(g, g, indexing='ij')
        pts = np.stack([lr.ravel(), lo.ravel()], 1)
        return np.exp(
            self.kde.score_samples(pts) - self._kde_max
        ).reshape(resolution, resolution)


# ==============================================================================
# CLI — Export یک instance برای مقایسه با GAMS
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export one LRP instance for GAMS comparison')
    parser.add_argument('--size',   default='lrp8',
                        choices=list(SIZE_CONFIGS.keys()),
                        help='سایز مسئله  (پیش‌فرض: lrp8)')
    parser.add_argument('--seed',   type=int, default=42,
                        help='Seed برای بازتولیدپذیری  (پیش‌فرض: 42)')
    parser.add_argument('--csv',    default='CA_AV_Collision_2019-2024.csv',
                        help='مسیر فایل داده تصادفات')
    parser.add_argument('--outdir', default='data',
                        help='پوشه خروجی  (پیش‌فرض: data/)')
    a = parser.parse_args()

    cfg = SIZE_CONFIGS[a.size]
    print(f'\n{"="*55}')
    print(f'LRP Instance Export   size={a.size}  seed={a.seed}')
    print(f'  n_cust={cfg["n_cust"]}, n_fac={cfg["n_fac"]}')
    print(f'{"="*55}')

    args_dict = {
        'n_cust':                   cfg['n_cust'],
        'n_fac':                    cfg['n_fac'],
        'capacity_per_compartment': cfg['capacity_per_comp'],
        'num_compartments':         cfg['num_comp'],
        'batch_size': 1,
        'test_size':  1,
    }

    gen = LRPDataGenerator(
        args=args_dict,
        collision_csv_path=a.csv,
        size_tag=a.size)

    tag = gen.export_instance_for_gams(seed=a.seed, out_dir=a.outdir)

    # نمایش task params برای استفاده در train/infer
    tp = task_specific_params_for_size(a.size)
    print(f'\n[Task Params — برای آموزش / inference]')
    for k, v in tp._asdict().items():
        print(f'  {k}: {v}')
