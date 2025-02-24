#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <chrono>



struct NeuronNode {
    int id;
    int type;    // SWC类型字段
    double x, y, z;
    double r;
    int parent;
};

struct NeuronTree {
    std::vector<NeuronNode> nodes;
    std::unordered_map<int, size_t> id_map;
};

struct NeuronDistSimple {
    double dist_12_allnodes = 0;
    double dist_21_allnodes = 0;
    double dist_allnodes = 0;
    double dist_apartnodes = 0;
    double percent_apartnodes = 0;
};

// 精确几何计算函数
struct Point3D { double x, y, z; };

double dist_L2(const Point3D& a, const Point3D& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
}

double dist_pt_to_line_seg(const Point3D& pt, const Point3D& p1, const Point3D& p2) {
    if (p1.x == p2.x && p1.y == p2.y && p1.z == p2.z) {
        return std::min(dist_L2(pt, p1), dist_L2(pt, p2));
    }    
      
    
    Point3D vec = {p2.x - p1.x, p2.y - p1.y, p2.z - p1.z};
    Point3D t_vec = {pt.x - p1.x, pt.y - p1.y, pt.z - p1.z};
    
    
    
    double t = (vec.x * t_vec.x + vec.y * t_vec.y + vec.z * t_vec.z) / 
               (vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    t = std::max(0.0, std::min(1.0, t));
    
    
    
    Point3D proj = {
        p1.x + vec.x * t,
        p1.y + vec.y * t,
        p1.z + vec.z * t
    };
    return dist_L2(pt, proj);
}

// 安全读取SWC文件
NeuronTree read_swc(const std::string& path) {
    NeuronTree tree;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        NeuronNode node;
        if (!(iss >> node.id >> node.type >> node.x >> node.y >> node.z >> node.r >> node.parent)) {
            continue;
        }
        if (node.parent > node.id)
        	std::cerr << "Parent After This!" << std::endl;
        tree.nodes.push_back(node);
        tree.id_map[node.id] = tree.nodes.size() - 1;
    }
    
    for (auto& node : tree.nodes) {
		        if (node.parent != -1 && !tree.id_map.count(node.parent)) {
		            std::cerr << "Invalid parent " << node.parent 
		                      << " for node " << node.id << std::endl;
		            node.parent = -1; // 将无效父节点标记为根节点
		        }
		    }
    return tree;
}


// 核心计算逻辑（双向）
void compute_directional(
    const NeuronTree& src, 
    const NeuronTree& target,
    double d_thres,
    double& sum, 
    size_t& nseg,
    double& sum_big,
    size_t& nseg_big
) {
    sum = 0; 
    nseg = 0;
    sum_big = 0;
    nseg_big = 0;
    
    
    std::vector<NeuronNode> sorted_target(target.nodes);
	    std::sort(sorted_target.begin(), sorted_target.end(), 
	        [](const NeuronNode& a, const NeuronNode& b){return a.id < b.id;});
    

    for (const auto& node : src.nodes) {
        if (node.parent == -1) continue;
        auto parent_it = src.id_map.find(node.parent);
        if (parent_it == src.id_map.end()) continue;
        
        const NeuronNode& parent = src.nodes[parent_it->second];
        Point3D p1 = {parent.x, parent.y, parent.z};
        Point3D p2 = {node.x, node.y, node.z};
        double seg_len = dist_L2(p1, p2);
        
        // 在compute_directional中添加
		if (seg_len < 1e-6) {
		    std::cerr << "Skip zero-length segment" << std::endl;
		    continue;
		}
        
		
		int steps = std::max(1, static_cast<int>(1 + seg_len + 0.5));
        
        for (int i = 0; i < steps; ++i) {
            double t = (steps > 1) ? static_cast<double>(i)/(steps-1) : 0.0;  // 关键修改
            Point3D pt = {
                p1.x * (1-t) + p2.x * t,
                p1.y * (1-t) + p2.y * t,
                p1.z * (1-t) + p2.z * t
            };
            
            double min_dist = std::numeric_limits<double>::max();
            //for (const auto& t_node : target.nodes) {
            for (const auto& t_node : sorted_target) {
                if (t_node.parent == -1) continue;
                auto t_parent_it = target.id_map.find(t_node.parent);
                if (t_parent_it == target.id_map.end()) continue;
                
                const NeuronNode& t_parent = target.nodes[t_parent_it->second];
                Point3D q1 = {t_parent.x, t_parent.y, t_parent.z};
                Point3D q2 = {t_node.x, t_node.y, t_node.z};
                double d = dist_pt_to_line_seg(pt, q1, q2);
                min_dist = std::min(min_dist, d);
            }
            
            if (min_dist != std::numeric_limits<double>::max()) {
                sum += min_dist;
                nseg++;
                if (min_dist >= d_thres) {
                    sum_big += min_dist;
                    nseg_big++;
                }
            }
        }
    }
}

NeuronDistSimple compute_scores(const NeuronTree& nt1, const NeuronTree& nt2, double d_thres) {
    NeuronDistSimple scores;
    
    // 计算双向指标
    double sum12, sum21;
    size_t nseg1, nseg2;
    double sum12_big, sum21_big;
    size_t nseg1_big, nseg2_big;
    
    compute_directional(nt1, nt2, d_thres, sum12, nseg1, sum12_big, nseg1_big);
    compute_directional(nt2, nt1, d_thres, sum21, nseg2, sum21_big, nseg2_big);

    // 处理除零错误
    scores.dist_12_allnodes = (nseg1 > 0) ? (sum12 / nseg1) : 0;
    scores.dist_21_allnodes = (nseg2 > 0) ? (sum21 / nseg2) : 0;
    scores.dist_allnodes = (scores.dist_12_allnodes + scores.dist_21_allnodes) / 2.0;
    
    // ==== 修复 DSA 计算逻辑 ====
    double dsa1 = (nseg1_big > 0) ? (sum12_big / nseg1_big) : 0;
    double dsa2 = (nseg2_big > 0) ? (sum21_big / nseg2_big) : 0;
    
    if (nseg1_big > 0 && nseg2_big > 0) {
        scores.dist_apartnodes = (dsa1 + dsa2) / 2.0;
    } else if (nseg1_big > 0) {
        scores.dist_apartnodes = dsa1;
    } else if (nseg2_big > 0) {
        scores.dist_apartnodes = dsa2;
    } else {
        scores.dist_apartnodes = 0;
    }
    
    // 计算 PDS
    double percent1 = (nseg1 > 0) ? (static_cast<double>(nseg1_big)/nseg1) : 0;
    double percent2 = (nseg2 > 0) ? (static_cast<double>(nseg2_big)/nseg2) : 0;
    scores.percent_apartnodes = (percent1 + percent2) / 2.0 * 100.0;
    
    return scores;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <swc1> <swc2> [d_thres=2.0]\n";
        return 1;
    }
    
    double d_thres = 2.0;
    if (argc >= 4) {
        try {
            d_thres = std::stod(argv[3]);
        } catch (...) {
            std::cerr << "Invalid threshold, using default 2.0\n";
        }
    }
    
    NeuronTree nt1 = read_swc(argv[1]);
    NeuronTree nt2 = read_swc(argv[2]);
    
    if (nt1.nodes.empty() || nt2.nodes.empty()) {
        std::cerr << "Error: Empty SWC file\n";
        return 1;
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    NeuronDistSimple scores = compute_scores(nt1, nt2, d_thres);
    
    auto total_end = std::chrono::high_resolution_clock::now();
	double total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000.0;
	
    
    std::cout.precision(6);
    std::cout << std::fixed
              << "ESA12: " << scores.dist_12_allnodes << "\n"
              << "ESA21: " << scores.dist_21_allnodes << "\n"
              << "ESA_Mean: " << scores.dist_allnodes << "\n"
              << "DSA: " << scores.dist_apartnodes << "\n"
              << "PDS: " << scores.percent_apartnodes << "%\n";
    std::cout.precision(0);
    std::cout << std::fixed<< "Total time: " << total_time << " ms\n";
    
    return 0;
}
