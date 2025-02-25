import math
import time
import numba
import numpy as np
from typing import List, Dict, Tuple

# 禁用Numba的类型检查以提升性能
numba.config.TYPECHECK_PRETTY = 1

# 使用Numpy结构化数组存储线段数据
dtype_node = np.dtype([
    ('id', np.int32),
    ('type', np.int32),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('r', np.float64),
    ('parent', np.int32)
])

dtype_segment = np.dtype([
    ('p1x', np.float64),
    ('p1y', np.float64),
    ('p1z', np.float64),
    ('p2x', np.float64),
    ('p2y', np.float64),
    ('p2z', np.float64)
])

@numba.njit(fastmath=True, cache=True)
def dist_L2(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

@numba.njit(fastmath=True, cache=True)
def dist_pt_to_line_seg_numba(pt, p1, p2):
    if (p1[0] == p2[0]) and (p1[1] == p2[1]) and (p1[2] == p2[2]):
        return min(dist_L2(pt, p1), dist_L2(pt, p2))
    
    vec_x = p2[0] - p1[0]
    vec_y = p2[1] - p1[1]
    vec_z = p2[2] - p1[2]
    
    t_vec_x = pt[0] - p1[0]
    t_vec_y = pt[1] - p1[1]
    t_vec_z = pt[2] - p1[2]
    
    denominator = vec_x**2 + vec_y**2 + vec_z**2
    if abs(denominator) < 1e-12:
        return min(dist_L2(pt, p1), dist_L2(pt, p2))
    
    t = (vec_x * t_vec_x + vec_y * t_vec_y + vec_z * t_vec_z) / denominator
    t = max(0.0, min(1.0, t))
    
    proj_x = p1[0] + vec_x * t
    proj_y = p1[1] + vec_y * t
    proj_z = p1[2] + vec_z * t
    
    dx = pt[0] - proj_x
    dy = pt[1] - proj_y
    dz = pt[2] - proj_z
    
    return math.sqrt(dx*dx + dy*dy + dz*dz)

@numba.njit(parallel=True, fastmath=True, cache=True)
def compute_directional_numba(
    src_segments,  # shape: (N, 2, 3)
    target_segments # shape: (M, 2, 3)
):
    total_sum = 0.0
    total_count = 0
    sum_big = 0.0
    count_big = 0
    d_thres = 2.0  # 暂时固定，后续动态传入
    
    # 预处理目标线段到连续内存
    target_array = np.ascontiguousarray(target_segments)
    
    for i in numba.prange(src_segments.shape[0]):
        seg = src_segments[i]
        p1 = seg[0]
        p2 = seg[1]
        
        seg_len = dist_L2(p1, p2)
        if seg_len < 1e-6:
            continue
        
        steps = max(1, int(1 + seg_len + 0.5))
        local_sum = 0.0
        local_count = 0
        local_sum_big = 0.0
        local_count_big = 0
        
        for j in range(steps):
            t = j / (steps - 1) if steps > 1 else 0.0
            pt = (
                p1[0] * (1 - t) + p2[0] * t,
                p1[1] * (1 - t) + p2[1] * t,
                p1[2] * (1 - t) + p2[2] * t
            )
            
            min_dist = np.inf
            for k in range(target_array.shape[0]):
                t_seg = target_array[k]
                q1 = t_seg[0]
                q2 = t_seg[1]
                d = dist_pt_to_line_seg_numba(pt, q1, q2)
                if d < min_dist:
                    min_dist = d
            
            if min_dist != np.inf:
                local_sum += min_dist
                local_count += 1
                if min_dist >= d_thres:
                    local_sum_big += min_dist
                    local_count_big += 1
        
        total_sum += local_sum
        total_count += local_count
        sum_big += local_sum_big
        count_big += local_count_big
    
    return total_sum, total_count, sum_big, count_big

class NeuronTreeFast:
    def __init__(self, path):
        start = time.time()
        self.segments = self._parse_swc(path)
        self.duration = time.time() - start
    
    def _parse_swc(self, path):
        nodes = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    node = (
                        int(parts[0]),      # id
                        int(parts[1]),      # type
                        float(parts[2]),    # x
                        float(parts[3]),    # y
                        float(parts[4]),    # z
                        float(parts[5]),    # r
                        int(parts[6])       # parent
                    )
                    nodes.append(node)
                except:
                    continue
        
        # 建立父子关系
        node_dict = {n[0]: n for n in nodes}
        segments = []
        for node in nodes:
            parent_id = node[6]
            if parent_id == -1:
                continue
            parent = node_dict.get(parent_id)
            if not parent:
                continue
            p1 = (parent[2], parent[3], parent[4])
            p2 = (node[2], node[3], node[4])
            segments.append((p1, p2))
        
        # 转换为Numba优化需要的数组格式 (N, 2, 3)
        seg_array = np.zeros((len(segments), 2, 3), dtype=np.float64)
        for i, (p1, p2) in enumerate(segments):
            seg_array[i, 0] = p1
            seg_array[i, 1] = p2
        return seg_array

def compute_neuron_metrics_fast(swc1_path, swc2_path, d_thres=2.0):
    """优化版本的计算函数"""
    # 读取并预处理数据
    t0 = time.time()
    nt1 = NeuronTreeFast(swc1_path)
    nt2 = NeuronTreeFast(swc2_path)
    io_time = time.time() - t0
    
    # 计算方向距离
    t_comp = time.time()
    sum12, nseg1, sum12_big, nseg1_big = compute_directional_numba(
        nt1.segments, nt2.segments
    )
    sum21, nseg2, sum21_big, nseg2_big = compute_directional_numba(
        nt2.segments, nt1.segments
    )
    comp_time = time.time() - t_comp
    
    # 计算结果（与之前相同）
    esa12 = sum12 / nseg1 if nseg1 else 0.0
    esa21 = sum21 / nseg2 if nseg2 else 0.0
    esa_mean = (esa12 + esa21) / 2
    
    dsa1 = sum12_big / nseg1_big if nseg1_big else 0.0
    dsa2 = sum21_big / nseg2_big if nseg2_big else 0.0
    dsa = (dsa1 + dsa2) / 2 if nseg1_big and nseg2_big else dsa1 + dsa2
    
    pds1 = (nseg1_big / nseg1 * 100) if nseg1 else 0.0
    pds2 = (nseg2_big / nseg2 * 100) if nseg2 else 0.0
    pds = (pds1 + pds2) / 2
    
    return {
        'ESA12': esa12,
        'ESA21': esa21,
        'ESA_mean': esa_mean,
        'DSA': dsa,
        'PDS': pds,
        'timing': {
            'io': io_time,
            'computation': comp_time,
            'total': io_time + comp_time
        }
    }

# 使用示例
if __name__ == "__main__":
    result = compute_neuron_metrics_fast("SLP_PN_neuron.swc", "SLP_PN_neuron.tif_x132_y681_z42_app2.swc",2.0)
    
    print(f"ESA12: {result['ESA12']:.6f}")
    print(f"ESA21: {result['ESA21']:.6f}")
    print(f"ESA_mean: {result['ESA_mean']:.6f}")
    print(f"DSA: {result['DSA']:.6f}")
    print(f"PDS: {result['PDS']:.6f}%")
    print(f"\nTime: {result['timing']['total']:.2f}s "
          f"(IO: {result['timing']['io']:.2f}s, "
          f"Compute: {result['timing']['computation']:.2f}s)")
