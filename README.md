
# Cell-Level Timing MVP (Nangate45 → ASAP7)

## Quick start

1) Put files:
```
data/
  lib/
    Nangate45/NangateOpenCellLibrary_typical.lib
    ASAP7/asap7sc6t_INVBUF_LVT_TT_nldm_211010.lib
    ASAP7/asap7sc6t_SIMPLE_LVT_TT_nldm_211010.lib
  spi/
    Nangate45/INVX1.spi...   
    ASAP7/asap7sc6t_26_L_211010.sp        
```

2) Build datasets:
```
python3 scripts/build_dataset.py --src_lib data/lib/Nangate45 --tgt_lib data/lib/ASAP7 --src_spi data/spi/Nangate45 --tgt_sp data/spi/ASAP7 --out_dir output --target_label_ratio 1.0
```

3) Baseline train (MLP):
```
 python3 scripts/train_mlp.py   --data_dir output   --mode joint   --epochs 50   --batch 256   --lr 0.001   --device cuda
```

4) HGAT-enabled train（HGAT）:
```
 python3 scripts/train_hgat.py   --data_dir output   --save_dir output   --tgt_spice data/spi/ASAP7/asap7sc6t_26_L_211010.sp   --s1_epochs 100   --s2_epochs 100   --lr 1e-3   --auto_lr   --lr_patience 8   --lr_factor 0.5   --min_lr 1e-6   --early_patience 30   --grad_clip 1.0
```

5) Evaluate:
```
 python3 scripts/eval_hgat.py --data_dir output --ckpt output/ckpt_transfer_best.pt --tgt_spice data/spi/ASAP7/asap7sc6t_26_L_211010.sp
 python3 scripts/eval_mlp.py   --data_dir output   --csv_name tgt_test.csv   --ckpt_name mlp_ckpt.pt
 ```


