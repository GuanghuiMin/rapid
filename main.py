#!/usr/bin/env python3
import argparse
import datetime
import gc
import os
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 全局 dtype 定义
DTYPE = np.float16

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman']

from scipy import sparse
from scipy.stats import kendalltau
from networks import generate_graph, load_real_data
from simulations.mc import run_monte_carlo_simulations
from simulations.pds import pds_prob
from simulations.rapid import rapid_prob
from simulations.dmp import dmp_prob
from eval import calculate_top_k_overlap
from utils import save_args_to_csv, calculate_centralities_approx, file_title_mapping
from sklearn.metrics import precision_score, recall_score, f1_score


# -------------------------------
# Helper functions
# -------------------------------
def binarize(probs, threshold):
    return (probs >= threshold).astype(np.int8)


def normalize_scores(score_dict):
    values = np.array(list(score_dict.values()), dtype=DTYPE)
    min_val, max_val = values.min(), values.max()
    if max_val == min_val:
        return {node: DTYPE(0.5) for node in score_dict}
    return {n: ((v - min_val) / (max_val - min_val)).astype(DTYPE) for n, v in score_dict.items()}


def calculate_error_bars_for_metrics(mc_list, method_probs, k):
    kendall_list, topk_list = [], []
    for mc in mc_list:
        tau, _ = kendalltau(mc.astype(np.float32), method_probs.astype(np.float32))
        kendall_list.append(tau)
        topk_list.append(calculate_top_k_overlap(method_probs.astype(np.float32), mc.astype(np.float32), k))
    return {
        "Kendall Tau": f"{np.mean(kendall_list):.4f} ± {np.std(kendall_list):.4f}",
        "Top-K Overlap": f"{np.mean(topk_list):.4f} ± {np.std(topk_list):.4f}"
    }


def calculate_normalized_error_with_errorbar(mc_list, method_probs):
    mse_list, mae_list = [], []
    for mc in mc_list:
        e = method_probs - mc
        mse_list.append(np.mean((e * e).astype(np.float32)))
        mae_list.append(np.mean(np.abs(e).astype(np.float32)))
    return np.mean(mse_list), np.std(mse_list), np.mean(mae_list), np.std(mae_list)


def calculate_mc_final_s_statistics(mc_s_trials):
    means = [float(a.mean()) for a in mc_s_trials]
    lowers = [float(np.percentile(a.astype(np.float32), 2.5)) for a in mc_s_trials]
    uppers = [float(np.percentile(a.astype(np.float32), 97.5)) for a in mc_s_trials]
    return (
        np.mean(means), np.mean(lowers), np.mean(uppers),
        np.std(means), np.std(lowers), np.std(uppers)
    )


def run_single_mc_trial(graph, beta, gamma, seeds, num_sims, idx, tmp):
    start = time.time()
    all_s, all_lens = [], []
    N = len(graph)
    probs = np.zeros(N, dtype=DTYPE)
    bs = min(args.mc_batch_size, num_sims)
    nb = (num_sims + bs - 1) // bs

    for b in range(nb):
        sz = min(bs, num_sims - b * bs)
        batch_res, batch_trajs, _ = run_monte_carlo_simulations(graph, beta, gamma, seeds, sz)

        for traj in batch_trajs:
            all_s.append(np.array(traj[-1], dtype=DTYPE))
            all_lens.append(len(traj))

        for sim in batch_res:
            st = sim[-1]['status']
            probs += np.array([DTYPE(1) if st[i] != 0 else DTYPE(0) for i in range(N)], dtype=DTYPE)

        del batch_res, batch_trajs
        gc.collect()

    probs /= DTYPE(num_sims)
    all_s_arr = np.array(all_s, dtype=DTYPE)
    del all_s
    gc.collect()
    np.save(os.path.join(tmp, f"mc_final_s_trial_{idx}.npy"), all_s_arr)
    np.save(os.path.join(tmp, f"mc_infection_probs_trial_{idx}.npy"), probs)

    return time.time() - start, float(np.mean(all_lens))


def run_method_trial(fn, graph, beta, gamma, seeds, tol, idx, tmp, *args2):
    start = time.time()
    res = fn(graph, beta, gamma, seeds, tol, *args2)
    return {
        "runtime": time.time() - start,
        "iterations": len(res),
        "final_state": res[-1]
    }


def sparse_dict_to_matrix(d, N):
    rows, cols, data = [], [], []
    if 'iteration' in d:
        d.pop('iteration')
    mp = {'susceptible': 0, 'infected': 1, 'recovered': 2}
    for st, m in d.items():
        for n, v in m.items():
            rows.append(int(n))
            cols.append(mp[st])
            data.append(float(v))
    arr = np.array(data, dtype=np.float32)
    return sparse.csr_matrix((arr, (rows, cols)), shape=(N, 3), dtype=np.float32)


def matrix_to_dict(M):
    sts = ['susceptible', 'infected', 'recovered']
    r = {s: {} for s in sts}
    coo = M.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        r[sts[j]][str(i)] = float(v)
    return r


def load_method_final_probs(idx, tmp, N):
    M = sparse.load_npz(os.path.join(tmp, f"method_results_trial_{idx}.npz"))
    d = matrix_to_dict(M)
    p = np.zeros(N, dtype=DTYPE)
    for st in ("infected", "recovered"):
        for k, v in d[st].items():
            p[int(k)] += DTYPE(v)
    del M, d
    gc.collect()
    return p


def get_infection_prob_1_minus_s(state, N):
    p = np.zeros(N, dtype=DTYPE)
    for k, v in state['susceptible'].items():
        p[int(k)] = DTYPE(1) - DTYPE(v)
    return p


def plot_threshold_curves_with_mc_horizontal_band(mc_frac, pds, dmp, lp, N, out, ths, fn):
    mc_mean = float(mc_frac.mean())
    mc_lo = float(np.percentile(mc_frac.astype(np.float32), 2.5))
    mc_hi = float(np.percentile(mc_frac.astype(np.float32), 97.5))
    t = np.linspace(0, 1, ths, dtype=DTYPE)

    def curve(x):
        return np.array([(x >= θ).mean() for θ in t], dtype=DTYPE)

    plt.figure(figsize=(8, 6))
    plt.fill_between(t, [mc_lo] * ths, [mc_hi] * ths, alpha=0.3,
                     label=f"MC-50 2.5%-97.5%: [{mc_lo:.3f},{mc_hi:.3f}]")
    plt.plot(t, [mc_mean] * ths, 'k-', lw=4, label=f"MC-50 mean={mc_mean:.3f}")
    if pds is not None: plt.plot(t, curve(pds), 'b-', lw=3, label="DMP(No Cavity)")
    if dmp is not None: plt.plot(t, curve(dmp), 'r--', lw=3, label="DMP")
    if lp is not None: plt.plot(t, curve(lp), 'g-.', lw=3, label="rapid")
    plt.axvline(x=float(args.threshold), ls='--', label=f"thr={args.threshold:.2f}")
    plt.xlabel("Threshold");
    plt.ylabel("Infected fraction")
    plt.grid();
    plt.legend()
    plt.title(file_title_mapping.get(fn, ""))
    plt.savefig(out.replace(".png", ".pdf"), dpi=600, bbox_inches='tight')
    plt.close()

    # -------------------------------


def main(args):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    od = os.path.join(args.output_dir, ts)
    td = os.path.join(od, "temp")
    os.makedirs(td, exist_ok=True)

    if args.file_name:
        G = load_real_data(args.data_dir, args.file_name)
    else:
        G = generate_graph(graph_type=args.graph_type,
                           avg_degree=args.average_degree,
                           nodes=args.nodes)
    if G.is_multigraph(): G = nx.DiGraph(G)
    N = G.number_of_nodes()
    args.nodes = N;
    args.average_degree = G.number_of_edges() / N

    S0 = max(round(args.initial_infected_ratio * N), 1)
    seeds = random.sample(list(G.nodes()), S0)
    save_args_to_csv(args, od)

    if args.beta > 0.1:
        args.gamma = args.gamma / args.beta * 0.1
        args.beta = 0.1

    mc50_probs, mc50_s, rt50, it50 = [], [], [], []
    for t in range(3):
        r, i = run_single_mc_trial(G, args.beta, args.gamma,
                                   seeds, args.num_simulations, t, td)
        rt50.append(r);
        it50.append(i)
        arr_p = np.load(os.path.join(td, f"mc_infection_probs_trial_{t}.npy")).astype(DTYPE)
        mc50_probs.append(arr_p)
        arr_s = np.load(os.path.join(td, f"mc_final_s_trial_{t}.npy")).astype(DTYPE)
        mc50_s.append(arr_s)
        del arr_p, arr_s
        gc.collect()

    mean_s, lb, ub, mean_s_err, lb_err, ub_err = calculate_mc_final_s_statistics(mc50_s)

    mean_rt50, std_rt50 = np.mean(rt50), np.std(rt50)
    mean_it50, std_it50 = np.mean(it50), np.std(it50)

    pd.DataFrame({
        "Metric": [
            "Mean S",
            "Lower (2.5%)",
            "Upper (97.5%)",
            "MC-50 Runtime (s)",
            "MC-50 Iterations"
        ],
        "Value": [
            f"{mean_s:.4f}",
            f"{lb:.4f}",
            f"{ub:.4f}",
            f"{mean_rt50:.4f} ± {std_rt50:.4f}",
            f"{mean_it50:.2f} ± {std_it50:.2f}"
        ]
    }).to_csv(os.path.join(od, "mc50_summary.csv"), index=False)

    baselines = {}
    for name, num in [("MC-5", 5), ("MC-10", 10)]:
        blist, rts, its = [], [], []
        for t in range(3):
            r, i = run_single_mc_trial(G, args.beta, args.gamma,
                                       seeds, num, t, td)
            rts.append(r);
            its.append(i)
            arr = np.load(os.path.join(td, f"mc_infection_probs_trial_{t}.npy")).astype(DTYPE)
            blist.append(arr)
        mean_p = np.mean(np.stack(blist), axis=0).astype(DTYPE)
        del blist
        gc.collect()
        errbar = calculate_error_bars_for_metrics(mc50_probs, mean_p, round(0.1 * N))
        mse, smse, mae, smae = calculate_normalized_error_with_errorbar(mc50_probs, mean_p)
        pd.DataFrame([{
            "Runtime (s)": f"{np.mean(rts):.4f} ± {np.std(rts):.4f}",
            "Iterations": f"{np.mean(its):.2f} ± {np.std(its):.2f}",
            "Kendall Tau": errbar["Kendall Tau"],
            "Top-K Overlap": errbar["Top-K Overlap"],
            "Normalized MSE": f"{mse:.4f} ± {smse:.4f}",
            "Normalized MAE": f"{mae:.4f} ± {smae:.4f}"
        }]).to_csv(os.path.join(od, f"{name}_summary.csv"), index=False)
        baselines[name] = {"probs": mean_p}
        gc.collect()

    methods = {
        "DMP(No Cavity)": pds_prob,
        "DMP": dmp_prob,
        "RAPID": rapid_prob
    }
    method_final_states = {}
    for name, fn in methods.items():
        plist, rts, its = [], [], []
        succeeded = 0
        for t in range(3):
            try:
                info = run_method_trial(fn, G, args.beta, args.gamma,
                                        seeds, args.tol, t, td,
                                        *( [args.p] if name=="RAPID" else [] ))
            except MemoryError as e:
                print(f"[WARN] {name} trial {t} OOM, skipping. {e}")
                continue
            succeeded += 1
            rts.append(info["runtime"])
            its.append(info["iterations"])
            p = get_infection_prob_1_minus_s(info["final_state"], N).astype(DTYPE)
            plist.append(p)

            M = sparse_dict_to_matrix(info["final_state"], N)
            sparse.save_npz(os.path.join(td, f"method_results_trial_{t}.npz"), M)
            del M
            gc.collect()

        if succeeded == 0:
            print(f"[ERROR] All 3 trials for method '{name}' failed. Skipping method.")
            continue

        mean_p = np.mean(np.stack(plist), axis=0).astype(DTYPE)
        method_final_states[name] = info["final_state"]

        errbar = calculate_error_bars_for_metrics(mc50_probs, mean_p, round(0.1*N))
        mse, smse, mae, smae = calculate_normalized_error_with_errorbar(mc50_probs, mean_p)
        pd.DataFrame([{
            "Runtime (s)": f"{np.mean(rts):.4f} ± {np.std(rts):.4f}",
            "Iterations": f"{np.mean(its):.2f} ± {np.std(its):.2f}",
            "Kendall Tau": errbar["Kendall Tau"],
            "Top-K Overlap": errbar["Top-K Overlap"],
            "Normalized MSE": f"{mse:.4f} ± {smse:.4f}",
            "Normalized MAE": f"{mae:.4f} ± {smae:.4f}"
        }]).to_csv(os.path.join(od, f"{name.replace(' ', '_')}_summary.csv"), index=False)

    # —— precision/recall/F1 —— #
    thr = args.threshold
    gt = binarize(np.mean(np.stack(mc50_probs), axis=0).astype(DTYPE), thr)
    rows = []
    for name in list(baselines) + list(methods.keys()):
        if name in baselines:
            probs = baselines[name]["probs"]
        else:
            probs = get_infection_prob_1_minus_s(method_final_states[name], N).astype(DTYPE)
        pred = binarize(probs, thr)
        rows.append({
            "Method": name,
            "Precision": precision_score(gt.astype(np.int8), pred.astype(np.int8), zero_division=0),
            "Recall": recall_score(gt.astype(np.int8), pred.astype(np.int8), zero_division=0),
            "F1": f1_score(gt.astype(np.int8), pred.astype(np.int8), zero_division=0)
        })
    pd.DataFrame(rows).to_csv(os.path.join(od, "precision_summary.csv"), index=False)

    # —— centralities —— #
    cent, cent_times = calculate_centralities_approx(G)
    pd.DataFrame(list(cent_times.items()), columns=["Centrality", "Time(s)"]) \
        .to_csv(os.path.join(od, "centrality_times.csv"), index=False)

    mat = {"Kendall Tau": {}, "Top-K Overlap": {}, "Precision": {}, "Recall": {}, "F1 Score": {}}

    for nm, vals in cent.items():
        arr = np.array([vals[i] for i in range(N)])
        m = calculate_error_bars_for_metrics(mc50_probs, arr, round(args.top_k_ratio * N))
        mat["Kendall Tau"][nm] = m["Kendall Tau"]
        mat["Top-K Overlap"][nm] = m["Top-K Overlap"]
        norm_arr = (arr - arr.min()) / (np.ptp(arr) + 1e-12)
        scores = (norm_arr >= args.threshold).astype(int)
        mat["Precision"][nm] = f"{precision_score(gt, scores):.4f}"
        mat["Recall"][nm] = f"{recall_score(gt, scores):.4f}"
        mat["F1 Score"][nm] = f"{f1_score(gt, scores):.4f}"
    pd.DataFrame(mat).T.to_csv(os.path.join(od, "centralities.csv"))

    for fname in os.listdir(td):
        os.remove(os.path.join(td, fname))
    os.rmdir(td)
    print(f"[Done] results in {od}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SIR sims with MC-5/10/50, DMP, RAPID and centralities"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--graph_type", type=str, default="")
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--average_degree", type=float, default=20)
    parser.add_argument("--beta", type=float, default=1 / 18)
    parser.add_argument("--gamma", type=float, default=1 / 9)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--mc_batch_size", type=int, default=5)
    parser.add_argument("--initial_infected_ratio", type=float, default=0.01)
    parser.add_argument("--top_k_ratio", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=1.3)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--file_name", type=str, default="soc-sign-bitcoinalpha.csv.gz")
    parser.add_argument("--output_dir", type=str, default="./output_0428")
    args = parser.parse_args()
    main(args)
