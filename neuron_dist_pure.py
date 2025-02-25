import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import collections

@dataclass
class NeuronNode:
    id: int
    type: int
    x: float
    y: float
    z: float
    r: float
    parent: int

class NeuronTree:
    def __init__(self):
        self.nodes: List[NeuronNode] = []
        self.id_map: Dict[int, int] = {}

Timers = collections.namedtuple('Timers', [
    'total', 
    'io', 
    'compute_12',
    'compute_21'
])

Point3D = collections.namedtuple('Point3D', ['x', 'y', 'z'])

def dist_L2(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def dist_pt_to_line_seg(pt: Point3D, p1: Point3D, p2: Point3D) -> float:
    # 与C++完全一致的实现（移除稳定系数）
    if p1 == p2:
        return min(dist_L2(pt, p1), dist_L2(pt, p2))
    
    vec = Point3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
    t_vec = Point3D(pt.x - p1.x, pt.y - p1.y, pt.z - p1.z)
    
    denominator = vec.x**2 + vec.y**2 + vec.z**2
    if abs(denominator) < 1e-12:  # 与C++处理方式完全一致
        return min(dist_L2(pt, p1), dist_L2(pt, p2))
    
    t = (vec.x * t_vec.x + vec.y * t_vec.y + vec.z * t_vec.z) / denominator
    t = max(0.0, min(1.0, t))
    
    proj = Point3D(
        p1.x + vec.x * t,
        p1.y + vec.y * t,
        p1.z + vec.z * t
    )
    return dist_L2(pt, proj)

def read_swc(path: str) -> NeuronTree:
    start = time.perf_counter()
    tree = NeuronTree()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 7:
                continue
            
            try:
                node = NeuronNode(
                    id=int(parts[0]),
                    type=int(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    r=float(parts[5]),
                    parent=int(parts[6])
                )
            except ValueError:
                continue
            
            # 与C++一致的parent校验
            if node.parent > node.id:
                pass  # 保持与C++相同的警告逻辑但静默
            
            tree.nodes.append(node)
            tree.id_map[node.id] = len(tree.nodes) - 1
    
    # 保持与C++相同的parent重置逻辑
    for node in tree.nodes:
        if node.parent != -1 and node.parent not in tree.id_map:
            node.parent = -1
    
    return tree

def compute_directional(
    src: NeuronTree, 
    target: NeuronTree, 
    d_thres: float
) -> Tuple[float, int, float, int]:
    sum_dist = 0.0
    n_seg = 0
    sum_big = 0.0
    n_seg_big = 0
    
    # 保持与C++完全一致的线段遍历顺序
    sorted_target = sorted(target.nodes, key=lambda n: n.id)
    
    for node in src.nodes:
        if node.parent == -1:
            continue
        parent_idx = src.id_map.get(node.parent)
        if parent_idx is None:
            continue
        parent = src.nodes[parent_idx]
        
        p1 = Point3D(parent.x, parent.y, parent.z)
        p2 = Point3D(node.x, node.y, node.z)
        seg_len = dist_L2(p1, p2)
        
        # 与C++完全一致的步骤数计算方法
        steps = max(1, int(1 + seg_len + 0.5))  # 关键修正点
        
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            pt = Point3D(
                p1.x * (1 - t) + p2.x * t,
                p1.y * (1 - t) + p2.y * t,
                p1.z * (1 - t) + p2.z * t
            )
            
            min_dist = float('inf')
            # 保持与C++一致的线段遍历顺序
            for t_node in sorted_target:
                if t_node.parent == -1:
                    continue
                t_parent_idx = target.id_map.get(t_node.parent)
                if t_parent_idx is None:
                    continue
                t_parent = target.nodes[t_parent_idx]
                
                q1 = Point3D(t_parent.x, t_parent.y, t_parent.z)
                q2 = Point3D(t_node.x, t_node.y, t_node.z)
                d = dist_pt_to_line_seg(pt, q1, q2)
                if d < min_dist:  # 直接比较，避免浮点误差
                    min_dist = d
            
            if min_dist != float('inf'):
                sum_dist += min_dist
                n_seg += 1
                if min_dist >= d_thres:
                    sum_big += min_dist
                    n_seg_big += 1
    
    return sum_dist, n_seg, sum_big, n_seg_big

def compute_neuron_metrics(swc1_path: str, swc2_path: str, d_thres: float = 2.0) -> Tuple[dict, Timers]:
    """计算神经元距离指标（带计时）
    
    返回值: (结果字典, 时间统计)
    """
    timers = {
        'total_start': time.perf_counter(),
        'io_start': time.perf_counter()
    }
    
    nt1 = read_swc(swc1_path)
    nt2 = read_swc(swc2_path)
    timers['io'] = time.perf_counter() - timers['io_start']
    
    if not nt1.nodes or not nt2.nodes:
        raise ValueError("Input SWC files cannot be empty")
    
    # 计算方向1
    timers['compute_12_start'] = time.perf_counter()
    sum12, nseg1, sum12_big, nseg1_big = compute_directional(nt1, nt2, d_thres)
    timers['compute_12'] = time.perf_counter() - timers['compute_12_start']
    
    # 计算方向2
    timers['compute_21_start'] = time.perf_counter()
    sum21, nseg2, sum21_big, nseg2_big = compute_directional(nt2, nt1, d_thres)
    timers['compute_21'] = time.perf_counter() - timers['compute_21_start']
    
    # 结果计算
    esa12 = sum12 / nseg1 if nseg1 else 0.0
    esa21 = sum21 / nseg2 if nseg2 else 0.0
    esa_mean = (esa12 + esa21) / 2
    
    dsa1 = sum12_big / nseg1_big if nseg1_big else 0.0
    dsa2 = sum21_big / nseg2_big if nseg2_big else 0.0
    dsa = (dsa1 + dsa2) / 2 if nseg1_big and nseg2_big else dsa1 + dsa2  # 修正逻辑
    
    pds1 = (nseg1_big / nseg1 * 100) if nseg1 else 0.0
    pds2 = (nseg2_big / nseg2 * 100) if nseg2 else 0.0
    pds = (pds1 + pds2) / 2
    
    # 时间统计
    total_time = time.perf_counter() - timers['total_start']
    time_stats = Timers(
        total=total_time * 1000,  # 毫秒
        io=timers['io'] * 1000,
        compute_12=timers['compute_12'] * 1000,
        compute_21=timers['compute_21'] * 1000
    )
    
    return {
        'ESA12': esa12,
        'ESA21': esa21,
        'ESA_mean': esa_mean,
        'DSA': dsa,
        'PDS': pds
    }, time_stats

if __name__ == "__main__":
    result, timers = compute_neuron_metrics("SLP_PN_neuron.swc", "SLP_PN_neuron.tif_x132_y681_z42_app2.swc", 2.0)
    
    print("Metrics:")
    print(f"ESA12: {result['ESA12']:.6f}")
    print(f"ESA21: {result['ESA21']:.6f}")
    print(f"ESA_mean: {result['ESA_mean']:.6f}")
    print(f"DSA: {result['DSA']:.6f}")
    print(f"PDS: {result['PDS']:.6f}%")
    
    print("\nTiming (ms):")
    print(f"Total: {timers.total:.1f} ms")
    print(f"  I/O: {timers.io:.1f} ms")
    print(f"  Compute12: {timers.compute_12:.1f} ms")
    print(f"  Compute21: {timers.compute_21:.1f} ms")
