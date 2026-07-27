#!/usr/bin/env python3
# ============================================================================
# fix_conv1d.py — 把 ONNX 里的 Conv1d 改写成等价 Conv2d (高=1)
#
# 用途: 绕开 ORT QNN EP 布局变换器对 Conv1d 的 NHWC bug
#   ("Node ... com.ms.internal.nhwc was inserted ... but was not selected
#    by that EP ... graph is now invalid")
# 原理: Conv1d(k,s,p) == Reshape->Conv2d((1,k),(1,s),(0,p,0,p))->Reshape,
#   数学完全等价, 数值不变; 权重经 Reshape 节点转 4D, 不改 model.data。
#
# 用法 (Termux, 需要 pip install onnx):
#   python fix_conv1d.py <输入目录> <输出目录>
# 例:
#   python fix_conv1d.py /sdcard/w8a16/encoder /sdcard/w8a16/encoder_fixed
# 输出目录里得到 model.onnx + model.data, 把它喂给 qnn_ctx_compile。
# ============================================================================

import os
import shutil
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def make_const(name, values):
    return numpy_helper.from_array(
        np.array(values, dtype=np.int64), name=name
    )


def main():
    if len(sys.argv) != 3:
        print(f"用法: {sys.argv[0]} <输入目录> <输出目录>")
        sys.exit(1)
    src_dir, dst_dir = sys.argv[1], sys.argv[2]
    os.makedirs(dst_dir, exist_ok=True)

    src_model = os.path.join(src_dir, "model.onnx")
    print(f"[1/4] 加载 {src_model} (3GB 级, 要等一会)...")
    model = onnx.load(src_model)  # 外置数据同目录自动跟随
    g = model.graph

    new_nodes = []
    new_inits = []
    n_fixed = 0
    uid = 0

    for node in g.node:
        if node.op_type != "Conv":
            new_nodes.append(node)
            continue
        ks = None
        for a in node.attribute:
            if a.name == "kernel_shape":
                ks = list(a.ints)
        if ks is None or len(ks) != 1:
            new_nodes.append(node)  # 已是 2D/3D conv, 不动
            continue

        # ---- 取 1D conv 参数 ----
        k = ks[0]
        strides, pads, dilations = [1], [0, 0], [1]
        for a in node.attribute:
            if a.name == "strides":
                strides = list(a.ints)
            elif a.name == "pads":
                pads = list(a.ints)
            elif a.name == "dilations":
                dilations = list(a.ints)
        s, d = strides[0], dilations[0]
        p_begin = pads[0]
        p_end = pads[1] if len(pads) > 1 else pads[0]

        X, W = node.input[0], node.input[1]
        B = node.input[2] if len(node.input) > 2 else ""
        Y = node.output[0]
        uid += 1
        tag = f"conv1d_fix_{uid}"

        # X4 = Reshape(X, [0,0,1,-1]) : (1,C,L) -> (1,C,1,L)
        sh_x = make_const(f"{tag}_shx", [0, 0, 1, -1])
        new_inits.append(sh_x)
        x4 = f"{tag}_x4"
        new_nodes.append(helper.make_node(
            "Reshape", [X, sh_x.name], [x4], name=f"{tag}_rx"))

        # W4 = Reshape(W, [0,0,1,k]) : (C',C,k) -> (C',C,1,k)
        sh_w = make_const(f"{tag}_shw", [0, 0, 1, k])
        new_inits.append(sh_w)
        w4 = f"{tag}_w4"
        new_nodes.append(helper.make_node(
            "Reshape", [W, sh_w.name], [w4], name=f"{tag}_rw"))

        # Y4 = Conv2d(X4, W4, B)
        y4 = f"{tag}_y4"
        conv_inputs = [x4, w4] + ([B] if B else [])
        new_nodes.append(helper.make_node(
            "Conv", conv_inputs, [y4], name=f"{tag}_conv2d",
            kernel_shape=[1, k], strides=[1, s],
            pads=[0, p_begin, 0, p_end], dilations=[1, d]))

        # Y = Reshape(Y4, [0,0,-1]) : (1,C',1,L') -> (1,C',L')
        sh_y = make_const(f"{tag}_shy", [0, 0, -1])
        new_inits.append(sh_y)
        new_nodes.append(helper.make_node(
            "Reshape", [y4, sh_y.name], [Y], name=f"{tag}_ry"))

        n_fixed += 1
        print(f"  改写: {node.name or Y}  Conv1d(k={k},s={s}) -> Conv2d")

    print(f"[2/4] 共改写 {n_fixed} 个 Conv1d")
    if n_fixed == 0:
        print("没有 Conv1d, 不用改, 直接退出")
        sys.exit(0)

    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)

    print(f"[3/4] 校验图...")
    onnx.checker.check_model(model, full_check=True)

    dst_model = os.path.join(dst_dir, "model.onnx")
    print(f"[4/4] 保存到 {dst_model} (外置数据 model.data)...")
    onnx.save_model(
        model, dst_model,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,  # 小于 1KB 的张量(新增的形状常量)内嵌进 .onnx,
                              # 大权重仍走外置; 权重未改时 model.data 与原文件一致
    )
    print("[done] 完成, 用输出目录喂 qnn_ctx_compile")


if __name__ == "__main__":
    main()
