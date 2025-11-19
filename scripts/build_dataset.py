
import argparse, os, json
import numpy as np
import pandas as pd
from parse_lib import parse_inv_arc_A_Y_auto
from spi2graph import parse_transistors_spice, extract_wl_features

def bilinear_interp(x_grid, y_grid, Z, x_new, y_new):
    X, Y = np.array(x_grid), np.array(y_grid)
    Z = np.array(Z)
    xv, yv = np.meshgrid(x_new, y_new, indexing="ij")
    out = np.zeros_like(xv, dtype=np.float64)
    for i in range(len(x_new)):
        for j in range(len(y_new)):
            x, y = x_new[i], y_new[j]
            ix = np.searchsorted(X, x); iy = np.searchsorted(Y, y)
            ix0 = max(0, min(ix-1, len(X)-2)); ix1 = ix0 + 1
            iy0 = max(0, min(iy-1, len(Y)-2)); iy1 = iy0 + 1
            x0, x1 = X[ix0], X[ix1]; y0, y1 = Y[iy0], Y[iy1]
            wx = 0.0 if x1==x0 else (x - x0)/(x1 - x0)
            wy = 0.0 if y1==y0 else (y - y0)/(y1 - y0)
            z00 = Z[ix0, iy0]; z10 = Z[ix1, iy0]; z01 = Z[ix0, iy1]; z11 = Z[ix1, iy1]
            out[i, j] = (1-wx)*(1-wy)*z00 + wx*(1-wy)*z10 + (1-wx)*wy*z01 + wx*wy*z11
    return out

def parse_spi_features_any(path):
    if (not path) or (not os.path.exists(path)):
        return {"wp_sum": 0.0, "wn_sum": 0.0, "wp_over_wn": 0.0}
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    devs = parse_transistors_spice(text)
    return extract_wl_features(devs)

def to_rows(tech, grid_slew, grid_cap, tables, extra_feats):
    rows = []
    vr = tables.get("nom_voltage", None)
    tr = tables.get("nom_temperature", None)
    for pol, M in [("rise", tables["cell_rise"]), ("fall", tables["cell_fall"])]:
        for i, s in enumerate(grid_slew):
            for j, c in enumerate(grid_cap):
                rows.append({
                    "tech": tech, "pol": pol, "slew": s, "cap": c,
                    "voltage": vr, "temp": tr,
                    "delay": M[i, j],
                    "wp_over_wn": extra_feats.get("wp_over_wn", 0.0),
                    "wp_sum": extra_feats.get("wp_sum", 0.0),
                    "wn_sum": extra_feats.get("wn_sum", 0.0),
                    "is_inv": 1, "stack_pu": 1, "stack_pd": 1
                })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_lib", required=True)
    ap.add_argument("--tgt_lib", required=True)
    ap.add_argument("--src_spi", default="")
    ap.add_argument("--tgt_spi", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grid_slew", type=int, default=12)
    ap.add_argument("--grid_cap", type=int, default=12)
    ap.add_argument("--target_label_ratio", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    S = parse_inv_arc_A_Y_auto(open(args.src_lib,"r",encoding="utf-8",errors="ignore").read())
    T = parse_inv_arc_A_Y_auto(open(args.tgt_lib,"r",encoding="utf-8",errors="ignore").read())

    smin = max(S["slew"].min(), T["slew"].min())
    smax = min(S["slew"].max(), T["slew"].max())
    cmin = max(S["cap"].min(),  T["cap"].min())
    cmax = min(S["cap"].max(),  T["cap"].max())
    grid_slew = np.linspace(smin, smax, args.grid_slew)
    grid_cap  = np.linspace(cmin, cmax, args.grid_cap)

    def interp_tables(TB):
        return {
            "cell_rise": bilinear_interp(TB["slew"], TB["cap"], TB["cell_rise"], grid_slew, grid_cap),
            "cell_fall": bilinear_interp(TB["slew"], TB["cap"], TB["cell_fall"], grid_slew, grid_cap),
            "rise_transition": bilinear_interp(TB["slew"], TB["cap"], TB["rise_transition"], grid_slew, grid_cap),
            "fall_transition": bilinear_interp(TB["slew"], TB["cap"], TB["fall_transition"], grid_slew, grid_cap),
            "nom_voltage": TB.get("nom_voltage", None),
            "nom_temperature": TB.get("nom_temperature", None),
        }

    Si = interp_tables(S); Ti = interp_tables(T)
    src_feats = parse_spi_features_any(args.src_spi)
    tgt_feats = parse_spi_features_any(args.tgt_spi)

    src_rows = to_rows("Nangate45", grid_slew, grid_cap, Si, src_feats)
    tgt_rows = to_rows("ASAP7",     grid_slew, grid_cap, Ti, tgt_feats)

    import pandas as pd, json
    df_src = pd.DataFrame(src_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_tgt = pd.DataFrame(tgt_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_label = max(1, int(len(df_tgt) * args.target_label_ratio))
    df_tgt_l = df_tgt.iloc[:n_label].copy()
    df_tgt_u = df_tgt.iloc[n_label:].copy()

    df_src.to_csv(os.path.join(args.out_dir, "src_delay.csv"), index=False)
    df_tgt_l.to_csv(os.path.join(args.out_dir, "tgt_delay_labeled.csv"), index=False)
    df_tgt_u.to_csv(os.path.join(args.out_dir, "tgt_delay_unlabeled.csv"), index=False)
    meta = {
        "grid_slew": grid_slew.tolist(), "grid_cap": grid_cap.tolist(),
        "n_src": len(df_src), "n_tgt_l": len(df_tgt_l), "n_tgt_u": len(df_tgt_u),
        "src_voltage": Si.get("nom_voltage"), "tgt_voltage": Ti.get("nom_voltage"),
        "src_temp": Si.get("nom_temperature"), "tgt_temp": Ti.get("nom_temperature"),
    }
    json.dump(meta, open(os.path.join(args.out_dir,"meta.json"),"w"), indent=2)
    print("Saved:", meta)

if __name__ == "__main__":
    main()
