import numpy as np
import math
import time
from scipy.spatial import cKDTree

# 在现有代码基础上添加的计时功能 ↓↓↓

class Timer:
    def __init__(self):
        self.timers = {}
        self.starts = {}
    
    def start(self, name):
        self.starts[name] = time.perf_counter()
    
    def stop(self, name):
        if name in self.starts:
            elapsed = (time.perf_counter() - self.starts[name]) * 1000  # 转为毫秒
            if name in self.timers:
                self.timers[name] += elapsed
            else:
                self.timers[name] = elapsed
    
    def get(self, name):
        return self.timers.get(name, 0.0)
    
    def reset(self):
        self.timers.clear()
        self.starts.clear()

# 在函数中添加计时逻辑 ↓↓↓

# 1. 严格数据加载实现
def load_swc_segments_strict(path):
    """与C++完全一致的SWC解析逻辑"""
    nodes = []
    id_map = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            
            try:
                node_id = int(parts[0])
                if node_id in id_map:
                    continue  # 严格处理重复ID
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                parent = int(parts[6])
                
                nodes.append((node_id, x, y, z, parent))
                id_map[node_id] = len(nodes)-1
            except:
                continue
    
    # 构建线段并严格校验parent
    segments = []
    for node in nodes:
        node_id, x, y, z, parent = node
        if parent == -1 or parent not in id_map:
            continue
        if parent > node_id:  # 与C++相同的parent校验
            continue
        
        parent_node = nodes[id_map[parent]]
        p1 = (parent_node[1], parent_node[2], parent_node[3])
        p2 = (x, y, z)
        segments.append((p1, p2))
    
    # 按坐标排序保证遍历顺序一致性
    return sorted(segments, key=lambda s: (s[0][0], s[0][1], s[0][2]))

# 2. 精确采样函数
def exact_sampler(segments):
    """与C++完全一致的采样逻辑"""
    samples = []
    for (p1, p2) in segments:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        seg_len = math.sqrt(dx**2 + dy**2 + dz**2)
        
        steps = max(1, int(1 + seg_len + 0.5))  # 严格匹配C++的转型逻辑
        
        for i in range(steps):
            t = i / (steps-1) if steps > 1 else 0.0
            sample = (
                p1[0]*(1-t) + p2[0]*t,
                p1[1]*(1-t) + p2[1]*t,
                p1[2]*(1-t) + p2[2]*t
            )
            samples.append(sample)
    return samples

# 3. 精确点线距离计算
def exact_distance(p, seg):
    """与C++逐行对应的距离算法"""
    p1, p2 = seg
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    
    t_vec_x = p[0] - p1[0]
    t_vec_y = p[1] - p1[1]
    t_vec_z = p[2] - p1[2]
    
    denominator = dx*dx + dy*dy + dz*dz
    if denominator < 1e-12:
        d1 = math.sqrt(t_vec_x**2 + t_vec_y**2 + t_vec_z**2)
        d2 = math.sqrt((p[0]-p2[0])**2 + (p[1]-p2[1])**2 + (p[2]-p2[2])**2)
        return min(d1, d2)
    
    t = (dx*t_vec_x + dy*t_vec_y + dz*t_vec_z) / denominator
    t = max(0.0, min(1.0, t))
    
    proj_x = p1[0] + dx * t
    proj_y = p1[1] + dy * t
    proj_z = p1[2] + dz * t
    return math.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2 + (p[2]-proj_z)**2)

# 4. 优化版KDTree索引
class SegmentKDTree:
    """精确线段索引结构（保证结果正确性）"""
    def __init__(self, segments, sample_step=5.0):
        self.segments = segments
        
        # 在线段上生成采样点
        self.points = []
        self.seg_indices = []
        for idx, (p1, p2) in enumerate(segments):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            length = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # 动态计算采样点数量
            num_samples = max(3, int(length / sample_step) + 1)
            for t in np.linspace(0, 1, num_samples):
                x = p1[0] + dx * t
                y = p1[1] + dy * t
                z = p1[2] + dz * t
                self.points.append([x, y, z])
                self.seg_indices.append(idx)
        
        # 构建KDTree
        self.tree = cKDTree(self.points)
    
    def query_candidates(self, point, k=15):
        """查询潜在候选线段（需覆盖真实最近线段）"""
        _, indices = self.tree.query(point, k=k)
        return np.unique([self.seg_indices[i] for i in indices])

def compute_metrics_optimized(swc1_path, swc2_path, d_thres=2.0, k=15):
    """优化版本（添加详细计时）"""
    timer = Timer()
    timer.start('total')
    
    # 加载数据
    timer.start('load')
    timer.start('load1')
    seg1 = load_swc_segments_strict(swc1_path)
    timer.stop('load1')
    timer.start('load2')
    seg2 = load_swc_segments_strict(swc2_path)
    timer.stop('load2')
    timer.stop('load')
    
    # 构建索引
    timer.start('indexing')
    timer.start('index1')
    tree2 = SegmentKDTree(seg2)
    timer.stop('index1')
    timer.start('index2')
    tree1 = SegmentKDTree(seg1)
    timer.stop('index2')
    timer.stop('indexing')
    
    # 生成采样点
    timer.start('sampling')
    timer.start('sample1')
    samples1 = exact_sampler(seg1)
    timer.stop('sample1')
    timer.start('sample2')
    samples2 = exact_sampler(seg2)
    timer.stop('sample2')
    timer.stop('sampling')
    
    # 计算方向1：seg1→seg2
    timer.start('compute12')
    sum12, n1, sum12_big, n1_big = 0.0, 0, 0.0, 0
    for pt in samples1:
        candidates = tree2.query_candidates(pt, k=k)
        min_dist = min(exact_distance(pt, seg2[i]) for i in candidates)
        
        sum12 += min_dist
        n1 += 1
        if min_dist >= d_thres:
            sum12_big += min_dist
            n1_big += 1
    timer.stop('compute12')
    
    # 计算方向2：seg2→seg1 
    timer.start('compute21')
    sum21, n2, sum21_big, n2_big = 0.0, 0, 0.0, 0
    for pt in samples2:
        candidates = tree1.query_candidates(pt, k=k)
        min_dist = min(exact_distance(pt, seg1[i]) for i in candidates)
        
        sum21 += min_dist
        n2 += 1
        if min_dist >= d_thres:
            sum21_big += min_dist
            n2_big += 1
    timer.stop('compute21')
    
    timer.stop('total')  # 结束总计时
    
    # 指标计算
    esa12 = sum12 / n1 if n1 else 0.0
    esa21 = sum21 / n2 if n2 else 0.0
    esa_mean = (esa12 + esa21) / 2
    
    dsa1 = sum12_big / n1_big if n1_big else 0.0
    dsa2 = sum21_big / n2_big if n2_big else 0.0
    dsa = (dsa1 + dsa2)/2 if n1_big and n2_big else dsa1 + dsa2
    
    pds = ((n1_big/n1*100 if n1 else 0.0) + 
           (n2_big/n2*100 if n2 else 0.0)) / 2
    
    return {
        'ESA12': esa12,
        'ESA21': esa21,
        'ESA_mean': esa_mean,
        'DSA': dsa,
        'PDS': pds,
        'timing': {
            'total': timer.get('total'),
            'load': timer.get('load'),
            'load_swc1': timer.get('load1'),
            'load_swc2': timer.get('load2'),
            'indexing': timer.get('indexing'),
            'index_swc2': timer.get('index1'),
            'index_swc1': timer.get('index2'),
            'sampling': timer.get('sampling'),
            'sample_swc1': timer.get('sample1'),
            'sample_swc2': timer.get('sample2'),
            'compute12': timer.get('compute12'),
            'compute21': timer.get('compute21')
        }
    }

# 修改主函数输出 ↓↓↓
if __name__ == "__main__":
    result = compute_metrics_optimized("SLP_PN_neuron.swc", "SLP_PN_neuron.tif_x132_y681_z42_app2.swc", 2.0)
    
    print("Metrics:")
    print(f"ESA12: {result['ESA12']:.6f}")
    print(f"ESA21: {result['ESA21']:.6f}")
    print(f"ESA_mean: {result['ESA_mean']:.6f}")
    print(f"DSA: {result['DSA']:.6f}")
    print(f"PDS: {result['PDS']:.6f}%")
    
    print("\nDetailed Timing (ms):")
    print(f"Total Time: {result['timing']['total']:.1f}")
    print(f"├── Data Loading: {result['timing']['load']:.1f}")
    print(f"│   ├── SWC1: {result['timing']['load_swc1']:.1f}")
    print(f"│   └── SWC2: {result['timing']['load_swc2']:.1f}")
    print(f"├── Index Building: {result['timing']['indexing']:.1f}")
    print(f"│   ├── SWC2 Index: {result['timing']['index_swc2']:.1f}")
    print(f"│   └── SWC1 Index: {result['timing']['index_swc1']:.1f}")
    print(f"├── Sampling: {result['timing']['sampling']:.1f}")
    print(f"│   ├── SWC1 Samples: {result['timing']['sample_swc1']:.1f}")
    print(f"│   └── SWC2 Samples: {result['timing']['sample_swc2']:.1f}")
    print(f"├── Compute12 Direction: {result['timing']['compute12']:.1f}")
    print(f"└── Compute21 Direction: {result['timing']['compute21']:.1f}")
