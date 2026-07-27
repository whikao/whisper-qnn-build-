#!/usr/bin/env python3
"""rank_probe.py — 探测 ONNX 模型里 rank>4 的张量和节点 (QNN 上限 rank-4)
不加载权重 (load_external_data=False), 秒级完成。
用法: python rank_probe.py <model.onnx>
"""
import sys
import onnx
from onnx import shape_inference

def main():
    path = sys.argv[1]
    print(f"加载 {path} (仅图结构)...")
    m = onnx.load(path, load_external_data=False)
    print("形状推导...")
    m = shape_inference.infer_shapes(m)

    # 收集所有张量形状: value_info + input + output
    shapes = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            shapes[vi.name] = [d.dim_value for d in t.shape.dim]

    over = {}
    for node in m.graph.node:
        for t in list(node.input) + list(node.output):
            sh = shapes.get(t)
            if sh and len(sh) > 4:
                over.setdefault(node.name or node.output[0], (node.op_type, []))
                over[node.name or node.output[0]][1].append((t, sh))

    print(f"\n=== rank>4 的节点: {len(over)} 个 ===")
    shown = 0
    for name, (op, tensors) in over.items():
        shown += 1
        if shown > 30:
            print(f"... 其余 {len(over)-30} 个省略")
            break
        print(f"\n{name}  [{op}]")
        for t, sh in tensors:
            print(f"    {t}  rank={len(sh)}  shape={sh}")

    # 统计各 rank 出现次数
    from collections import Counter
    cnt = Counter(len(sh) for sh in shapes.values() if len(sh) > 4)
    print(f"\n=== 超维张量 rank 分布: {dict(cnt)} ===")

if __name__ == "__main__":
    main()
