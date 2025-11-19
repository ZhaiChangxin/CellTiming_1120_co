
# Cell-Level Timing MVP (Nangate45 → ASAP7) — HGAT-ready

## Quick start

1) Put files:
```
data/
  lib/
    Nangate45/INVX1.lib
    ASAP7/INVX1.lib
  spi/
    Nangate45/INVX1.spi   # Nangate45 SPICE (optional for features/HGAT)
    ASAP7/INVX1.sp        # ASAP7 PEX/SPICE (optional for features/HGAT)
```

2) Build datasets:
```
python scripts/build_dataset.py --src_lib data/lib/Nangate45/INVX1.lib --tgt_lib data/lib/ASAP7/INVX1.lib   --src_spi data/spi/Nangate45/INVX1.spi --tgt_spi data/spi/ASAP7/INVX1.sp   --out_dir outputs --grid_slew 12 --grid_cap 12 --target_label_ratio 0.1
```

3) Baseline train (no HGAT):
```
python scripts/train.py --data_dir outputs --epochs 120 --batch 256 --lr 2e-3 --hid 128 --cmd_k 5
```

4) HGAT-enabled train（异构图，不引入路径注意力）:
```
python scripts/train.py --data_dir outputs --epochs 120 --batch 256 --lr 2e-3 --hid 128 --cmd_k 5   --use_hgat --src_spice data/spi/Nangate45/INVX1.spi --tgt_spice data/spi/ASAP7/INVX1.sp
```

5) Evaluate:
```
python scripts/eval.py --data_dir outputs --ckpt outputs/ckpt.pt
```

### Notes
- `.lib` 解析已适配 **Nangate45 (INV_X1)** 与 **ASAP7 (INVx1_ASAP7_6t_R)** 等命名差异；自动定位 `pin(Y/ZN) -> timing(related_pin="A")`。
- `.sp/.spi` 解析会抓取 M 行中 **W/L**（自动识别单位），并构建 **NET/PMOS/NMOS** 的最小异构图；HGAT 在训练时端到端更新。
- HGAT 需要安装 DGL（CPU/GPU 任一版本均可）。
