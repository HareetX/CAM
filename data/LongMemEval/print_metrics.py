import json
import argparse
import sys
from collections import defaultdict

def load_json(file_path):
    """读取JSON文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON 格式解析失败 '{file_path}': {e}")
        return None

def load_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSONL 格式解析失败 '{file_path}': {e}")
        return None

def load_output_data(file_path):
    """尝试读取 Output 文件，支持 JSON 和 JSONL 格式"""
    data = load_json(file_path)
    if data is not None:
        return data
    data = load_jsonl(file_path)
    if data is None:
        print(f"错误: 无法加载 Output 文件 '{file_path}'。请检查文件格式是否为 JSON 或 JSONL。")
        sys.exit(1)
    return data

def calculate_metrics(output_data, reference_data):
    """计算分组和总体指标"""
    # 1. 建立 ID 到 Question Type 的映射
    # Reference 文件中使用 'question_id' 和 'question_type'
    qid_to_type = {}
    for item in reference_data:
        qid = item.get('question_id')
        q_type = item.get('question_type', 'Unknown') # 如果没有类型，标记为 Unknown
        if qid:
            qid_to_type[qid] = q_type

    # 2. 初始化存储结构
    # 结构: { 'Type': { 'Metric': [val1, val2] } }
    metrics_storage = defaultdict(lambda: defaultdict(list))
    
    # 需要统计的指标键名 (Output 文件中的 key)
    target_metrics = ['LLM_Judge', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']

    matched_count = 0

    # 3. 遍历 Output 数据并聚合
    for item in output_data:
        qid = item.get('QID')
        if not qid:
            qid = item.get('question_id')  # 兼容不同命名
        
        # 如果 QID 缺失或不在 reference 中，跳过 (或者你可以选择归为 Unknown)
        if not qid or qid not in qid_to_type:
            continue
        
        q_type = qid_to_type[qid]
        matched_count += 1
        
        # 提取指标并存入对应分类和 Overall
        for metric in target_metrics:
            # 获取分数，默认为 0.0
            val = item.get(metric)
            if val is None:
                val = 0.0
                if metric == 'LLM_Judge':
                    val = item.get('autoeval_label', {}).get('label', 0.0)
                    
            
            # 添加到对应 Type
            metrics_storage[q_type][metric].append(val)
            # 添加到 Overall
            metrics_storage['Overall'][metric].append(val)

    return metrics_storage, matched_count, target_metrics

def print_table(metrics_storage, target_metrics):
    """打印格式化表格"""
    # 定义表头
    headers = ['Question Type', 'Count'] + target_metrics
    
    # 定义行格式: Type占30字符, Count占8字符, 指标各占12字符
    row_format = "{:<30} | {:<8} | " + " | ".join(["{:<12}"] * len(target_metrics))
    
    separator = "-" * (30 + 8 + 3 + (15 * len(target_metrics)))

    print(separator)
    print(row_format.format(*headers))
    print(separator)

    # 排序：按字母顺序排序类型，确保 'Overall' 在最后
    sorted_types = sorted([t for t in metrics_storage.keys() if t != 'Overall'])
    if 'Overall' in metrics_storage:
        sorted_types.append('Overall')

    for q_type in sorted_types:
        data = metrics_storage[q_type]
        # 获取样本数量 (假设所有指标样本数一致)
        count = len(data[target_metrics[0]]) if target_metrics else 0
        
        # 计算平均值并格式化为百分比
        avgs = []
        for metric in target_metrics:
            values = data[metric]
            avg = sum(values) / len(values) if values else 0.0
            # 格式化: 乘以100转为百分比，保留2位小数
            avgs.append(f"{avg*100:.2f}")

        print(row_format.format(q_type, count, *avgs))
    
    print(separator)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="计算并打印 QA 实验指标 (按 Question Type 分组)")
    parser.add_argument('output_file', help="Output 文件路径 (包含 QID 和 分数)")
    parser.add_argument('reference_file', help="Reference 文件路径 (包含 question_id 和 question_type)")
    
    args = parser.parse_args()

    # 加载数据
    output_data = load_output_data(args.output_file)
    reference_data = load_json(args.reference_file)

    # 计算
    metrics_storage, count, target_metrics = calculate_metrics(output_data, reference_data)

    # 输出结果
    if count == 0:
        print("\n[警告] 未在 Reference 文件中找到与 Output 文件匹配的 ID。")
        print("请检查 output 文件中的 'QID' 是否与 reference 文件中的 'question_id' 一致。")
    else:
        print(f"\n成功匹配样本数: {count}")
        print("\n实验结果 (平均分 %):")
        print_table(metrics_storage, target_metrics)

if __name__ == "__main__":
    main()