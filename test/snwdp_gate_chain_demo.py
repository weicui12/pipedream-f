# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation


def build_chain_network(
    N_gates=5,
    reach_len=20_000.0,
    bed_slope=1/25_000,
    n_manning=0.015,
    channel_width=25.0,
    Q0=50.0,
    h0=5.0,
    z_down=0.0
):
    # SJ0..SJ(2N)  共 2N+1 个节点
    n_sj = 2 * N_gates + 1
    sj_ids = np.arange(n_sj, dtype=int)

    # 底高程：reach（偶->奇）有坡降，gate（奇->偶）零长度
    drop = reach_len * bed_slope  # 0.8 m/段
    z_inv = np.zeros(n_sj, dtype=float)
    z_inv[-1] = z_down

    for i in range(n_sj - 2, -1, -1):
        is_reach = (i % 2 == 0) and ((i + 1) % 2 == 1)
        z_inv[i] = z_inv[i + 1] + (drop if is_reach else 0.0)

    superjunctions = pd.DataFrame({
        "id": sj_ids,
        "name": [f"SJ{i}" for i in sj_ids],
        "z_inv": z_inv,
        "h_0": np.full(n_sj, h0),
        "bc": np.array([False] * (n_sj - 1) + [True]),  # 仅最下游边界
        "storage": ["functional"] * n_sj,
        "a": np.zeros(n_sj),
        "b": np.zeros(n_sj),
        "c": np.full(n_sj, channel_width * 300.0),  # 等效水面面积（稳一点）
        "max_depth": np.full(n_sj, 50.0)
    })

    # 5段渠道：SJ(2k)->SJ(2k+1)
    sl_ids = np.arange(N_gates, dtype=int)
    superlinks = pd.DataFrame({
        "id": sl_ids,
        "name": [f"Reach{k+1}" for k in sl_ids],
        "sj_0": 2 * sl_ids,
        "sj_1": 2 * sl_ids + 1,
        "in_offset": np.zeros(N_gates),
        "out_offset": np.zeros(N_gates),
        "C_uk": np.full(N_gates, 0.9),
        "C_dk": np.full(N_gates, 0.9),

        "dx": np.full(N_gates, reach_len),
        "n": np.full(N_gates, n_manning),

        # 如果你这行报错，把 rect_open 改成 rect 或 rectangular 再试
        "shape": ["rect_open"] * N_gates,

        "g1": np.full(N_gates, channel_width),  # 矩形宽
        "g2": np.zeros(N_gates),
        "g3": np.zeros(N_gates),
        "g4": np.zeros(N_gates),

        "Q_0": np.full(N_gates, Q0),
        "h_0": np.full(N_gates, h0),
        "A_s": np.full(N_gates, channel_width * 300.0),
        "ctrl": np.full(N_gates, False),
        "A_c": np.zeros(N_gates),
        "C": np.zeros(N_gates)
    })

    # 5座闸：SJ(2k+1)->SJ(2k+2)
    o_ids = np.arange(N_gates, dtype=int)
    orifices = pd.DataFrame({
        "id": o_ids,
        "name": [f"Gate{k+1}" for k in o_ids],
        "sj_0": 2 * o_ids + 1,
        "sj_1": 2 * o_ids + 2,
        "orientation": ["bottom"] * N_gates,
        "C": np.full(N_gates, 0.7),
        "A": np.full(N_gates, 30.0),
        "y_max": np.full(N_gates, 3.0),
        "z_o": np.zeros(N_gates)
    })

    return superjunctions, superlinks, orifices


def chainage_from_id(sj_id, reach_len=20_000.0):
    return (sj_id // 2) * reach_len


def main():
    # 题设
    N = 5
    dt = 30.0
    t_end = 6 * 3600.0
    times = np.arange(0, t_end + dt, dt)

    # 建模
    superjunctions, superlinks, orifices = build_chain_network(N_gates=N)

    model = SuperLink(
        superlinks=superlinks,
        superjunctions=superjunctions,
        links=None,
        junctions=None,
        orifices=orifices
    )

    # 边界输入
    n_sj = len(superjunctions)
    Q_in = pd.DataFrame(0.0, index=times, columns=np.arange(n_sj))
    Q_in.iloc[:, 0] = 50.0  # 上游流量边界

    H_bc = pd.DataFrame(np.nan, index=times, columns=np.arange(n_sj))
    H_bc.iloc[:, -1] = 5.0  # 下游水位边界（z=0, h=5）

    # 闸开度：稳态验证先恒定
    u_o = np.full(N, 0.6, dtype=float)

    # 仿真并记录
    with Simulation(model, Q_in=Q_in, H_bc=H_bc) as sim:
        first = True
        while sim.t <= sim.t_end:
            sim.step(dt=dt, u_o=u_o, first_time=first)
            first = False
            sim.record_state()
            sim.print_progress()

    # Simulation.__exit__ already converts states to DataFrames with the
    # superjunction names as columns, so access it directly instead of
    # rebuilding from a raw dict (which lost the SJ* labels and caused the
    # KeyError).
    H_df = sim.states.H_j.sort_index()

    # 直接用记录的时间轴，保持与模拟输出一致
    t_rec = H_df.index.astype(float)
    H_df.index = t_rec

    # 列名映射成 SJ 名称
    id2name = superjunctions.set_index("id")["name"].to_dict()
    H_df = H_df.rename(columns=id2name)

    # 图：水位过程
    plt.figure()
    for col in H_df.columns:
        plt.plot(H_df.index / 3600.0, H_df[col], label=col)
    plt.xlabel("Time (hr)")
    plt.ylabel("Head H_j (m)")
    plt.title("Superjunction heads (串联多闸干线)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 末时刻水面线
    H_last = H_df.iloc[-1]
    prof = superjunctions[["id", "name", "z_inv"]].copy()
    prof["x"] = prof["id"].apply(lambda k: chainage_from_id(int(k)))
    prof["H"] = prof["name"].map(H_last.to_dict())
    prof["h"] = prof["H"] - prof["z_inv"]
    prof = prof.sort_values(["x", "id"])

    print("\n=== End-profile (last recorded timestep) ===")
    print(prof[["id", "name", "x", "z_inv", "H", "h"]].to_string(index=False))

    # 闸前后水头差
    print("\nGate head differences (H_up - H_dn):")
    for k in range(N):
        sj_up = f"SJ{2*k+1}"
        sj_dn = f"SJ{2*k+2}"
        dH = H_df[sj_up] - H_df[sj_dn]
        print(f"  Gate{k+1}: mean dH={dH.mean():.4f} m | min={dH.min():.4f} | max={dH.max():.4f}")


if __name__ == "__main__":
    main()
