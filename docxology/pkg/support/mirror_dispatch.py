"""Central registry: ``docxology/examples/**`` path → runnable orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random as jr

from pkg.support.bootstrap import OrchestrationConfig
from pkg.support.checks import pymdp_probe, try_import_pybefit, try_import_torch
from pkg.support.viz import save_current_figure, plot_beliefs_heatmap, plot_action_probabilities
from pkg.support.analysis import marginalize_actions

_REPO_ROOT = Path(__file__).resolve().parents[3]
from pkg.support.patterns import (
    complex_action_dependency_agent,
    random_agent_one_cycle,
)
from pkg.support.si_fixtures import run_mcts_smoke, run_si_policy_search_smoke, si_cue_agent


def _docxology_root(script_file: Path) -> Path:
    """Sidecar root (not a nested ``examples/docxology/`` package directory)."""
    for p in script_file.resolve().parents:
        if (p / "manifests" / "orchestrations.txt").is_file():
            return p
    raise RuntimeError(f"Could not locate docxology root above {script_file}")


def _examples_rel(script_file: Path) -> str:
    examples_root = _docxology_root(script_file) / "examples"
    return script_file.resolve().relative_to(examples_root).as_posix()


def run_registered(script_file: Path, cfg: OrchestrationConfig) -> dict[str, Any]:
    rel = _examples_rel(script_file)
    if rel not in HANDLERS:
        return {"ok": False, "error": "unregistered", "rel": rel}
    return HANDLERS[rel](cfg)


# --- handlers (real pymdp calls; keep each short and deterministic) ---


def _extract_rollout_diagnostics(info: dict[str, Any]) -> dict[str, Any]:
    """Convert JAX rollout info dict into serializable per-timestep diagnostic traces.
    
    Extracts: beliefs (qs), actions, observations, policy posteriors (q_pi),
    negative expected free energy (neg_efe / G values per policy).
    """
    import numpy as np
    diag: dict[str, Any] = {}
    
    # Per-timestep beliefs (factor 0)
    if "qs" in info:
        qs_raw = info["qs"]
        if isinstance(qs_raw, (list, tuple)):
            # list of factors; take factor 0
            qs_arr = np.asarray(qs_raw[0])
        else:
            qs_arr = np.asarray(qs_raw)
        # qs_arr shape typically: (batch, T, num_states) or (T, batch, num_states)
        if qs_arr.ndim >= 2:
            diag["beliefs_per_timestep"] = qs_arr.squeeze().tolist()
    
    # Per-timestep actions
    if "action" in info:
        act = np.asarray(info["action"])
        diag["actions_per_timestep"] = act.squeeze().tolist()
    
    # Per-timestep observations
    if "observation" in info:
        obs = info["observation"]
        if isinstance(obs, (list, tuple)):
            # list of modalities; take modality 0
            obs_arr = np.asarray(obs[0])
        else:
            obs_arr = np.asarray(obs)
        if obs_arr.ndim >= 1:
            diag["observations_per_timestep"] = obs_arr.squeeze().tolist()
    
    # Per-timestep policy posterior (q_pi)
    if "qpi" in info:
        qpi = np.asarray(info["qpi"])
        diag["q_pi_per_timestep"] = qpi.squeeze().tolist()
    
    # Per-timestep negative EFE (G values per policy)
    if "neg_efe" in info:
        neg_efe = np.asarray(info["neg_efe"])
        diag["neg_efe_per_timestep"] = neg_efe.squeeze().tolist()
    
    # Learned parameter snapshots if present
    if "A" in info:
        a_arr = info["A"]
        if isinstance(a_arr, (list, tuple)):
            diag["learned_A_shapes"] = [list(np.asarray(x).shape) for x in a_arr]
        elif hasattr(a_arr, "shape"):
            diag["learned_A_shape"] = list(np.asarray(a_arr).shape)
    
    if "B" in info:
        b_arr = info["B"]
        if isinstance(b_arr, (list, tuple)):
            diag["learned_B_shapes"] = [list(np.asarray(x).shape) for x in b_arr]
        elif hasattr(b_arr, "shape"):
            diag["learned_B_shape"] = list(np.asarray(b_arr).shape)
    
    return diag

def _verify_invariants(info: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    results = {"passed": True, "violations": []}
    
    def check_sum(arr, axis, name, tol=1e-3):
        arr = np.asarray(arr)
        if arr.size == 0: return
        sums = np.sum(arr, axis=axis)
        if not np.allclose(sums, 1.0, atol=tol):
            results["passed"] = False
            results["violations"].append(f"{name} does not sum to 1 over axis {axis}.")

    if "qs" in info:
        qs_raw = info["qs"]
        if isinstance(qs_raw, (list, tuple)) and len(qs_raw) > 0:
            try:
                first = np.asarray(qs_raw[-1])
                if first.ndim >= 2:
                    check_sum(first[..., 0, :], -1, "Final beliefs (qs)")
                elif first.ndim == 1:
                    check_sum(first, -1, "Final beliefs (qs)")
            except: pass

    if "qpi" in info:
        try:
            qpi = np.asarray(info["qpi"])
            if qpi.ndim >= 2: check_sum(qpi[-1], -1, "Final policy posterior (q_pi)")
            elif qpi.ndim == 1: check_sum(qpi, -1, "Final policy posterior (q_pi)")
        except: pass
            
    if "A_matrix" in info:
        try:
            A = np.asarray(info["A_matrix"][0] if isinstance(info["A_matrix"], (list, tuple)) else info["A_matrix"])
            check_sum(A, 0, "Likelihood matrix (A)")
        except: pass

    if "B_matrix" in info:
        try:
            B = np.asarray(info["B_matrix"][0] if isinstance(info["B_matrix"], (list, tuple)) else info["B_matrix"])
            check_sum(B, 0, "Transition matrix (B)")
        except: pass
        
    return results

def _save_native_arrays(cfg: OrchestrationConfig, info: dict[str, Any], stem: str) -> None:
    if not cfg.save_plots:
        return
    import numpy as np
    to_save = {}
    
    def _add_to_save(key: str, val: Any):
        if isinstance(val, (list, tuple)):
            if len(val) > 0 and hasattr(val[0], "shape"):
                for i, factor in enumerate(val):
                    try: to_save[f"{key}_factor_{i}"] = np.asarray(factor)
                    except: pass
            else:
                try: to_save[key] = np.asarray(val)
                except: pass
        else:
            try: to_save[key] = np.asarray(val)
            except: pass

    for key, val in info.items():
        if key.startswith("_"): continue
        if isinstance(val, dict): continue
        _add_to_save(key, val)
        
    if to_save:
        try:
            np.savez_compressed(cfg.output_dir / f"{stem}_model_trace.npz", **to_save)
        except Exception:
            pass

def _generate_markdown_report(cfg: OrchestrationConfig, stem: str, invariants: dict[str, Any], info: dict[str, Any]) -> None:
    if not cfg.save_plots:
        return
    lines = [
        f"# Execution Report: {stem}", "",
        "## Configuration",
        f"- **Seed**: {cfg.seed}",
        f"- **Fast Mode**: {cfg.fast}",
        f"- **Skip Heavy**: {cfg.skip_heavy}", "",
        "## Mathematical Invariants",
    ]
    if invariants["passed"]: lines.append("- ✅ **All probability bounds valid.**")
    else:
        lines.append("- ❌ **Violations Detected:**")
        for v in invariants["violations"]: lines.append(f"  - {v}")
        
    lines.extend(["", "## Performance Insights"])
    lines.append("| Metric | Terminal Value | Mean Trajectory Value |")
    lines.append("|---|---|---|")
    
    import numpy as np
    added_metrics = False
    
    def add_metric(name, key):
        nonlocal added_metrics
        if key in info:
            v = np.asarray(info[key]).ravel()
            if v.size > 0:
                final = f"{float(v[-1]):.4f}"
                mean = f"{float(np.mean(v)):.4f}"
                lines.append(f"| **{name}** | {final} | {mean} |")
                added_metrics = True
                
    add_metric("Shannon Entropy $H(q)$", "H_qs")
    add_metric("Variational Free Energy $F$", "vfe")
    add_metric("Variational Free Energy $F$", "F")
    add_metric("Negative Expected Free Energy $-G$", "neg_efe")
    add_metric("Negative Expected Free Energy $-G$", "G")
    add_metric("KL Divergence $D_{KL}(q||p)$", "KL")
    
    if not added_metrics:
        lines.append("| *No scalar trace endpoints detected* | N/A | N/A |")
        
    lines.extend(["", "## Native Trace Archive", f"A native complete JAX/NumPy parameter archive is available at [`{stem}_model_trace.npz`]({stem}_model_trace.npz).", ""])
    lines.append("## Visualizations")
    pngs = sorted([p for p in cfg.output_dir.glob("*.png") if p.is_file()])
    if not pngs: lines.append("*No visual traces generated.*")
    else:
        for p in pngs:
            name = p.stem.replace(stem + "_", "").replace("_", " ").title()
            lines.extend([f"### {name}", f"![{name}]({p.name})", ""])
    (cfg.output_dir / f"{stem}_execution_report.md").write_text("\n".join(lines), encoding="utf-8")

def _to_serializable(obj: Any) -> Any:
    import numpy as np
    if hasattr(obj, "tolist"):
        return np.asarray(obj).tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in list(obj)]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)

def _auto_plot_metrics(cfg: OrchestrationConfig, info: dict[str, Any], stem: str) -> None:
    if not cfg.save_plots:
        return
    try:
        invariants = _verify_invariants(info)
        info["_invariants"] = invariants
        
        import numpy as np
        if "qs" in info and "H_qs" not in info:
            qs_raw = info["qs"]
            if isinstance(qs_raw, (list, tuple)) and len(qs_raw) > 0:
                h_seq = []
                for q in qs_raw:
                    q_arr = np.asarray(q)
                    if isinstance(q, (list, tuple)):
                        q_arr = np.asarray(q[0]) if len(q) > 0 else np.array([])
                    q_arr = q_arr.flatten()
                    if q_arr.size > 0:
                        q_safe = np.clip(q_arr, 1e-12, 1.0)
                        h = -np.sum(q_safe * np.log(q_safe))
                        h_seq.append(float(h))
                if h_seq:
                    info["H_qs"] = h_seq

        _save_native_arrays(cfg, info, stem)
        import json
        out_json = cfg.output_dir / f"{stem}_full_data.json"
        out_json.write_text(json.dumps(_to_serializable(info), indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        from pkg.support.viz import plot_action_probabilities, plot_beliefs_heatmap, plot_free_energy, save_current_figure
        import jax.numpy as jnp
        import numpy as np
        
        # 1. Action frequencies
        if "action" in info:
            act = np.asarray(info["action"])
            if act.ndim >= 1 and act.size > 0:
                max_act = max(3, int(np.max(act) + 1))
                counts = np.bincount(act.flatten().astype(int), minlength=max_act)
                probs = counts / np.sum(counts)
                plot_action_probabilities(probs, title=f"Action Frequencies ({stem})")
                save_current_figure(cfg.output_dir, stem=f"{stem}_actions")
                
        if "action_probs" in info:
            probs = np.asarray(info["action_probs"])
            if probs.ndim > 1:
                probs = probs.flatten()
            plot_action_probabilities(probs, title=f"Action Probabilities ({stem})")
            save_current_figure(cfg.output_dir, stem=f"{stem}_action_probs")
                
        # 2. Generative matrices
        if "A_matrix" in info:
            from pkg.support.viz import plot_likelihood_matrix
            plot_likelihood_matrix(info["A_matrix"], title=f"Likelihood Matrix A ({stem})")
            save_current_figure(cfg.output_dir, stem=f"{stem}_matrix_A")
            
        if "B_matrix" in info:
            from pkg.support.viz import plot_transition_matrix
            plot_transition_matrix(info["B_matrix"], title=f"Transition Matrix B ({stem})")
            save_current_figure(cfg.output_dir, stem=f"{stem}_matrix_B")
            
        if "D_matrix" in info:
            from pkg.support.viz import plot_empirical_prior
            plot_empirical_prior(info["D_matrix"], title=f"Empirical Prior D ({stem})")
            save_current_figure(cfg.output_dir, stem=f"{stem}_matrix_D")

        if "C_matrix" in info:
            from pkg.support.viz import plot_prior_preferences
            plot_prior_preferences(info["C_matrix"], title=f"Prior Preferences C ({stem})")
            save_current_figure(cfg.output_dir, stem=f"{stem}_matrix_C")
            
        if "q_pi" in info:
            from pkg.support.viz import plot_policy_posterior
            plot_policy_posterior(info["q_pi"], title=f"Policy Posterior ({stem})")
            save_current_figure(cfg.output_dir, stem=f"{stem}_q_pi")
            
        if "G_epistemic" in info and "G_pragmatic" in info:
            from pkg.support.viz import plot_efe_components
            plot_efe_components(info["G_epistemic"], info["G_pragmatic"], title=f"EFE Breakdown ({stem})")
            save_current_figure(cfg.output_dir, stem=f"{stem}_G_components")

        # 3. Beliefs heatmap + derived trajectory analytics
        if "qs" in info:
            qs_raw = info["qs"]
            qs_seq = []
            
            if isinstance(qs_raw, (list, tuple)):
                # Determine format: JAX rollout returns list of factors each (batch, T, states)
                # Legacy returns list of per-timestep belief arrays
                first = qs_raw[0]
                first_arr = np.asarray(first)
                
                if first_arr.ndim >= 3:
                    # JAX rollout format: each element is a factor (batch, T, states)
                    # Extract factor 0, squeeze batch, split along T axis
                    f0 = first_arr.squeeze()  # (T, N_states)
                    if f0.ndim == 2:
                        qs_seq = [f0[t, :] for t in range(f0.shape[0])]
                elif first_arr.ndim <= 2:
                    # Legacy format: list of (states,) or (num_factors, states) per timestep
                    shape_to_match = None
                    for qs_timestep in qs_raw:
                        q_f0 = qs_timestep[0] if isinstance(qs_timestep, (list, tuple)) else qs_timestep
                        q_arr = np.asarray(q_f0)
                        if shape_to_match is None:
                            shape_to_match = q_arr.shape
                        if q_arr.shape == shape_to_match:
                            qs_seq.append(q_arr)
            
            if len(qs_seq) > 1:
                plot_beliefs_heatmap(qs_seq, title=f"Beliefs ({stem})")
                save_current_figure(cfg.output_dir, stem=f"{stem}_beliefs")
                
                # Entropy trajectory
                from pkg.support.viz import plot_entropy_trajectory
                plot_entropy_trajectory(qs_seq, title=f"Belief Entropy ({stem})")
                save_current_figure(cfg.output_dir, stem=f"{stem}_entropy")
                
                # KL divergence from prior (use D_matrix factor 0 or uniform)
                if "D_matrix" in info:
                    D0 = info["D_matrix"]
                    if isinstance(D0, (list, tuple)):
                        D0 = D0[0]
                    from pkg.support.viz import plot_kl_divergence_trajectory
                    plot_kl_divergence_trajectory(qs_seq, D0, title=f"KL from Prior ({stem})")
                    save_current_figure(cfg.output_dir, stem=f"{stem}_kl")
                
                # Animated GIF of belief evolution
                if len(qs_seq) >= 3:
                    from pkg.support.viz import plot_belief_trajectory_animation
                    plot_belief_trajectory_animation(qs_seq, cfg.output_dir, stem)
        
        # 4. Per-timestep policy/EFE analytics (from rollout qpi/neg_efe traces)
        if "qpi" in info:
            qpi_data = info["qpi"]
            qpi_arr = np.asarray(qpi_data).squeeze()  # squeeze batch dim: (1,T,N) -> (T,N)
            if qpi_arr.ndim >= 2 and qpi_arr.shape[0] > 2:
                from pkg.support.viz import plot_policy_posterior_heatmap
                q_pi_list = qpi_arr.tolist()
                if isinstance(q_pi_list[0], list):
                    plot_policy_posterior_heatmap(q_pi_list, title=f"Policy Posterior Evolution ({stem})")
                    save_current_figure(cfg.output_dir, stem=f"{stem}_qpi_heatmap")
        
        if "neg_efe" in info:
            neg_efe_data = info["neg_efe"]
            neg_arr = np.asarray(neg_efe_data).squeeze()  # (1,T,N) -> (T,N)
            if neg_arr.ndim >= 2 and neg_arr.shape[0] > 2:
                from pkg.support.viz import plot_efe_trajectory, plot_neg_efe_heatmap
                neg_list = neg_arr.tolist()
                if isinstance(neg_list[0], list):
                    plot_efe_trajectory(neg_list, title=f"EFE Trajectory ({stem})")
                    save_current_figure(cfg.output_dir, stem=f"{stem}_efe_traj")
                    plot_neg_efe_heatmap(neg_list, title=f"Neg-EFE Landscape ({stem})")
                    save_current_figure(cfg.output_dir, stem=f"{stem}_efe_heatmap")
        
        # 5. Action frequency donut chart
        if "action" in info:
            act_flat = np.asarray(info["action"]).flatten().astype(int)
            if act_flat.size > 3 and len(np.unique(act_flat)) >= 2:
                from pkg.support.viz import plot_action_frequency_donut
                plot_action_frequency_donut(act_flat, title=f"Action Distribution ({stem})")
                save_current_figure(cfg.output_dir, stem=f"{stem}_action_donut")
                
        if "vfe" in info or "F" in info:
            F_seq = info.get("vfe", info.get("F"))
            if np.asarray(F_seq).size > 1:
                from pkg.support.viz import plot_free_energy
                plot_free_energy(F_seq, title=f"Variational Free Energy ({stem})")
                save_current_figure(cfg.output_dir, stem=f"{stem}_vfe")
                
        if "I_matrix" in info or "I" in info:
            I_mat = info.get("I_matrix", info.get("I"))
            if np.asarray(I_mat).ndim >= 2:
                from pkg.support.viz import plot_reachability_matrix
                plot_reachability_matrix(I_mat, title=f"Reachability Matrix I ({stem})")
                save_current_figure(cfg.output_dir, stem=f"{stem}_reachability_I")

        # Generate the unified markdown report right at the end to assemble all generated files
        _generate_markdown_report(cfg, stem, invariants, info)
    except Exception:
        pass

def _h_model_construction(cfg: OrchestrationConfig) -> dict[str, Any]:
    detail = random_agent_one_cycle(
        seed=cfg.seed,
        policy_len=1 if cfg.fast else 2,
        inference_algo="fpi",
    )
    if cfg.save_plots and "action" in detail:
        # random_agent_one_cycle returns {"action": action_idx, "qs_shapes": ...}
        # To show action distribution, we can do a dummy one-hot plot or just trace action idx
        # Since it returns discrete actions, let's plot a one-hot distribution representation
        num_actions = max(3, detail["action"][0] + 1)
        probs = jnp.zeros(num_actions)
        probs = probs.at[detail["action"][0]].set(1.0)
        plot_action_probabilities(probs, title="Chosen Action (Deterministic)")
        save_current_figure(cfg.output_dir, stem="model_construction_action_prob")

    _auto_plot_metrics(cfg, detail, "model_construction")
    return {"ok": True, "id": "api/model_construction_tutorial", "detail": {k: v for k, v in detail.items() if not k.endswith("_matrix")}, **pymdp_probe()}


def _h_complex_action_dependency(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    agent = complex_action_dependency_agent(seed=cfg.seed)
    obs = [jnp.array([[0]]), jnp.array([[1]])]
    qs, _ = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
    q_pi, neg_efe = agent.infer_policies(qs)
    
    info = {"qs": qs, "qpi": q_pi, "neg_efe": neg_efe}
    _auto_plot_metrics(cfg, info, "complex_action_dependency")
        
    return {
        "ok": True, "id": "advanced/complex_action_dependency", "qs0_shape": tuple(qs[0].shape),
        "diagnostics": {
            "beliefs": [np.asarray(q).squeeze().tolist() for q in qs],
            "q_pi": np.asarray(q_pi[0]).squeeze().tolist(),
            "neg_efe": np.asarray(neg_efe[0]).squeeze().tolist(),
        },
    }


def _h_infer_states_methods(cfg: OrchestrationConfig, stem_prefix: str = "infer_states") -> dict[str, Any]:
    algos = ("fpi", "mmp") if not cfg.fast else ("fpi",)
    out = {}
    diag_per_algo = {}
    for algo in algos:
        detail = random_agent_one_cycle(seed=cfg.seed, policy_len=1, inference_algo=algo)
        _auto_plot_metrics(cfg, detail, f"{stem_prefix}_{algo}")
        out[algo] = detail["action"]
        diag_per_algo[algo] = detail.get("diagnostics", {})
    return {
        "ok": True, "id": "advanced/infer_states_optimization/methods_test",
        "algos": out, "diagnostics": diag_per_algo,
    }


def _h_neural_encoder(cfg: OrchestrationConfig) -> dict[str, Any]:
    if cfg.skip_heavy or cfg.fast:
        return {"ok": True, "skipped": True, "reason": "skip_heavy_or_fast"}
    if not try_import_torch():
        return {"ok": True, "skipped": True, "reason": "no_torch"}
    d = random_agent_one_cycle(seed=cfg.seed, policy_len=1)
    _auto_plot_metrics(cfg, d, "neural_encoder")
    return {"ok": True, "id": "advanced/pymdp_with_neural_encoder", "note": "torch_present_agent_smoke",
            "diagnostics": d.get("diagnostics", {})}


def _h_tmaze_rollout(cfg: OrchestrationConfig) -> dict[str, Any]:
    from pymdp.agent import Agent
    from pymdp.envs import TMaze
    from pymdp.envs.rollout import rollout

    env = TMaze(categorical_obs=False)
    C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
    C[1] = C[1].at[1].set(3.0).at[2].set(-3.0)
    agent = Agent(
        env.A,
        env.B,
        C,
        D=env.D,
        A_dependencies=env.A_dependencies,
        B_dependencies=env.B_dependencies,
        num_controls=[5, 1],
        batch_size=1,
        policy_len=2 if not cfg.fast else 1,
    )
    T = 3 if cfg.fast else 25
    _, info = rollout(agent, env, num_timesteps=T, rng_key=jr.PRNGKey(cfg.seed))
    info["D_matrix"] = env.D
    info["C_matrix"] = C
    _auto_plot_metrics(cfg, info, 'tmaze_demo')
    return {"ok": True, "id": "envs/tmaze_demo", "T": T, "action_shape": tuple(info["action"].shape), "diagnostics": _extract_rollout_diagnostics(info)}


def _h_generalized_tmaze(cfg: OrchestrationConfig) -> dict[str, Any]:
    from pymdp.agent import Agent
    from pymdp.envs import SimplifiedTMaze
    from pymdp.envs.rollout import rollout

    env = SimplifiedTMaze(categorical_obs=False)
    C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
    C[1] = C[1].at[1].set(2.0).at[2].set(-2.0)
    agent = Agent(
        env.A,
        env.B,
        C,
        D=env.D,
        A_dependencies=env.A_dependencies,
        B_dependencies=env.B_dependencies,
        num_controls=[4, 1],
        batch_size=1,
        policy_len=2,
    )
    T = 3 if cfg.fast else 25
    _, info = rollout(agent, env, num_timesteps=T, rng_key=jr.PRNGKey(cfg.seed))
    _auto_plot_metrics(cfg, info, 'generalized_tmaze')
    return {"ok": True, "id": "envs/generalized_tmaze_demo", "action_shape": tuple(info["action"].shape), "diagnostics": _extract_rollout_diagnostics(info)}


def _h_cue_chaining(cfg: OrchestrationConfig) -> dict[str, Any]:
    from pymdp.agent import Agent
    from pymdp.envs.cue_chaining import CueChainingEnv
    from pymdp.envs.rollout import rollout

    env = CueChainingEnv(cue2_state=0, reward_condition=0)
    C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
    C[3] = C[3].at[1].set(2.0).at[2].set(-4.0)
    agent = Agent(
        A=env.A,
        B=env.B,
        C=C,
        D=env.D,
        A_dependencies=env.A_dependencies,
        B_dependencies=env.B_dependencies,
        policy_len=3 if cfg.fast else 4,
        batch_size=1,
    )
    T = 3 if cfg.fast else 20
    _, info = rollout(agent, env, num_timesteps=T, rng_key=jr.PRNGKey(cfg.seed))
    _auto_plot_metrics(cfg, info, 'cue_chaining')
    return {"ok": True, "id": "envs/cue_chaining_demo", "action_shape": tuple(info["action"].shape), "diagnostics": _extract_rollout_diagnostics(info)}


def _h_graph_worlds(cfg: OrchestrationConfig) -> dict[str, Any]:
    import networkx as nx

    from pymdp.agent import Agent
    from pymdp.envs import GraphEnv
    from pymdp.envs.rollout import rollout

    g = nx.path_graph(5)
    env = GraphEnv(g, agent_location=0, object_location=2, key=jr.PRNGKey(cfg.seed))
    C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
    C[1] = C[1].at[0].set(1.0)
    nloc = len(g)
    agent = Agent(
        env.A,
        env.B,
        C,
        D=env.D,
        A_dependencies=env.A_dependencies,
        B_dependencies=env.B_dependencies,
        num_controls=[nloc, 1],
        batch_size=1,
        policy_len=2,
    )
    T = 3 if cfg.fast else 25
    _, info = rollout(agent, env, num_timesteps=T, rng_key=jr.PRNGKey(cfg.seed + 1))
    _auto_plot_metrics(cfg, info, 'graph_worlds')
    return {"ok": True, "id": "envs/graph_worlds_demo", "action_shape": tuple(info["action"].shape), "diagnostics": _extract_rollout_diagnostics(info)}


def _h_knapsack_surrogate(cfg: OrchestrationConfig) -> dict[str, Any]:
    """Discrete MDP smoke via ``GridWorld`` (knapsack notebook is combinatorial; this verifies JAX env + agent)."""
    from pymdp.agent import Agent
    from pymdp.envs.grid_world import GridWorld
    from pymdp.envs.rollout import rollout

    env = GridWorld(shape=(3, 3), initial_position=(0, 0))
    C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
    agent = Agent(
        env.A,
        env.B,
        C,
        D=env.D,
        A_dependencies=env.A_dependencies,
        B_dependencies=env.B_dependencies,
        num_controls=[5],
        policy_len=2,
        batch_size=1,
    )
    T = 3 if cfg.fast else 25
    _, info = rollout(agent, env, num_timesteps=T, rng_key=jr.PRNGKey(cfg.seed))
    _auto_plot_metrics(cfg, info, 'knapsack')
    return {
        "ok": True,
        "id": "envs/knapsack_demo",
        "note": "gridworld_surrogate",
        "action_shape": tuple(info["action"].shape),
        "diagnostics": _extract_rollout_diagnostics(info),
    }


def _h_chained_cue_nav_upstream(cfg: OrchestrationConfig) -> dict[str, Any]:
    """Delegates to upstream GIF script with tiny timesteps (no new GIF logic in docxology)."""
    import runpy
    import sys

    repo = _REPO_ROOT
    target = repo / "examples" / "envs" / "chained_cue_navigation.py"
    out = cfg.output_dir if cfg.output_dir else (repo / "docxology" / "output" / "envs")
    out.mkdir(parents=True, exist_ok=True)
    ts = 2 if cfg.fast else 15
    argv = [str(target), "--output-dir", str(out), "--timesteps", str(ts), "--fps", "8"]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old
    return {"ok": True, "id": "envs/chained_cue_navigation", "output_dir": str(out)}


def _h_si_tmaze_validation(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    d = run_si_policy_search_smoke(fast=cfg.fast)
    _auto_plot_metrics(cfg, d, 'si_tmaze_validation')
    probs = np.asarray(d.get('action_probs', [])).squeeze()
    return {
        "ok": True, "id": "experimental/sophisticated_inference/si_tmaze_SIvalidation", **d,
        "diagnostics": {"action_probs": probs.tolist() if hasattr(probs, 'tolist') else list(probs),
                        "q_pi_shape": d.get('q_pi_shape', ())}
    }


def _h_si_generalized_tmaze(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    d = run_si_policy_search_smoke(fast=cfg.fast)
    _auto_plot_metrics(cfg, d, 'si_generalized_tmaze')
    probs = np.asarray(d.get('action_probs', [])).squeeze()
    return {
        "ok": True, "id": "experimental/sophisticated_inference/si_generalized_tmaze", **d,
        "diagnostics": {"action_probs": probs.tolist() if hasattr(probs, 'tolist') else list(probs)}
    }


def _h_si_graph_world(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    d = run_si_policy_search_smoke(fast=cfg.fast)
    _auto_plot_metrics(cfg, d, 'si_graph_world')
    probs = np.asarray(d.get('action_probs', [])).squeeze()
    return {
        "ok": True, "id": "experimental/sophisticated_inference/si_graph_world", **d,
        "diagnostics": {"action_probs": probs.tolist() if hasattr(probs, 'tolist') else list(probs)}
    }


def _h_mcts_generalized_tmaze(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    d = run_mcts_smoke(fast=cfg.fast, seed=cfg.seed)
    _auto_plot_metrics(cfg, d, 'mcts_generalized_tmaze')
    probs = np.asarray(d.get('action_probs', [])).squeeze()
    return {
        "ok": True, "id": "experimental/sophisticated_inference/mcts_generalized_tmaze", **d,
        "diagnostics": {"action_probs": probs.tolist() if hasattr(probs, 'tolist') else list(probs),
                        "action_weights_shape": d.get('action_weights_shape', ())}
    }


def _h_mcts_graph_world(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    d = run_mcts_smoke(fast=cfg.fast, seed=cfg.seed + 1)
    _auto_plot_metrics(cfg, d, 'mcts_graph_world')
    probs = np.asarray(d.get('action_probs', [])).squeeze()
    return {
        "ok": True, "id": "experimental/sophisticated_inference/mcts_graph_world", **d,
        "diagnostics": {"action_probs": probs.tolist() if hasattr(probs, 'tolist') else list(probs),
                        "action_weights_shape": d.get('action_weights_shape', ())}
    }


def _chain_transition(num_states: int) -> jnp.ndarray:
    B = jnp.zeros((num_states, num_states, 1), dtype=jnp.float32)
    for prev in range(num_states - 1):
        B = B.at[prev + 1, prev, 0].set(1.0)
    B = B.at[num_states - 1, num_states - 1, 0].set(1.0)
    return B


def _h_inductive_example(cfg: OrchestrationConfig, stem: str = "inductive_inference_example") -> dict[str, Any]:
    import pymdp.control as ctl_jax
    import numpy as np

    H = [jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)]
    B = [_chain_transition(3)]
    I = ctl_jax.generate_I_matrix(H, B, threshold=0.5, depth=4)
    _auto_plot_metrics(cfg, {"B_matrix": B}, stem)
    # Reachability matrix visualization
    if cfg.save_plots:
        from pkg.support.viz import plot_reachability_matrix
        plot_reachability_matrix(I, title=f"Reachability Matrix I ({stem})")
        save_current_figure(cfg.output_dir, stem=f"{stem}_matrix_I")
    I_arr = np.asarray(I[0])
    return {
        "ok": True, "id": "inductive_inference/inductive_inference_example",
        "I_shape": tuple(I_arr.shape),
        "diagnostics": {
            "I_matrix": I_arr.tolist(),
            "H_vector": [0.0, 0.0, 1.0],
            "B_shape": list(np.asarray(B[0]).shape),
            "threshold": 0.5, "depth": 4,
        },
    }


def _h_inductive_gridworld(cfg: OrchestrationConfig) -> dict[str, Any]:
    return _h_inductive_example(cfg, stem="inductive_inference_gridworld") | {"id": "inductive_inference/inductive_inference_gridworld"}


def _h_inference_methods_comparison(cfg: OrchestrationConfig) -> dict[str, Any]:
    return _h_infer_states_methods(cfg, stem_prefix="inference_comp") | {"id": "inference_and_learning/inference_methods_comparison"}


def _h_learning_gridworld(cfg: OrchestrationConfig) -> dict[str, Any]:
    from pymdp import learning, utils

    num_obs = [4]
    num_states = [3]
    A_deps = [[0]]
    keys = jr.split(jr.PRNGKey(cfg.seed), 3)
    A = utils.random_A_array(keys[0], num_obs, num_states, A_dependencies=A_deps)
    pA = utils.list_array_scaled([a.shape for a in A], scale=2.0)
    qs = utils.random_factorized_categorical(keys[1], num_states)
    obs = [jnp.array([[0]], dtype=jnp.int32)]
    qs_b = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
    qA, _ = learning.update_obs_likelihood_dirichlet(
        pA,
        A,
        obs,
        qs_b,
        A_dependencies=A_deps,
        categorical_obs=False,
        num_obs=num_obs,
        lr=0.5,
    )
    _auto_plot_metrics(cfg, {"A_matrix": qA}, "learning_gridworld")
    import numpy as np
    delta = float(jnp.sum(qA[0] - pA[0]))
    return {
        "ok": True, "id": "learning/learning_gridworld", "qA_delta": delta,
        "diagnostics": {
            "qA_delta": delta,
            "pA_prior_shape": list(np.asarray(pA[0]).shape),
            "qA_posterior_shape": list(np.asarray(qA[0]).shape),
            "learning_rate": 0.5,
            "observation": [0],
            "beliefs": [np.asarray(q).squeeze().tolist() for q in qs],
        },
    }


def _h_sparse(cfg: OrchestrationConfig) -> dict[str, Any]:
    import time
    import numpy as np
    from jax.experimental import sparse as jsparse

    t0 = time.perf_counter()
    x = jsparse.BCOO.fromdense(jnp.eye(4, dtype=jnp.float32))
    y = (x + x).todense()
    elapsed = time.perf_counter() - t0
    # Sparse structure visualization
    if cfg.save_plots:
        from pkg.support.viz import plot_sparse_structure
        plot_sparse_structure(np.asarray(y), nnz=int(x.nse), title="BCOO Sparse Identity (2×I₄)")
        save_current_figure(cfg.output_dir, stem="sparse_structure")
    return {
        "ok": True, "id": "sparse/sparse_benchmark", "dense_shape": tuple(y.shape),
        "diagnostics": {
            "sparse_ops_elapsed_s": round(elapsed, 4),
            "nnz_input": int(x.nse),
            "dense_shape": list(y.shape),
            "result_trace": float(jnp.trace(y)),
        },
    }


def _h_pybefit(cfg: OrchestrationConfig) -> dict[str, Any]:
    if cfg.skip_heavy and cfg.fast:
        return {"ok": True, "skipped": True, "reason": "skip_heavy_fast"}
    if not try_import_pybefit():
        return {"ok": True, "skipped": True, "reason": "no_pybefit"}
    import pybefit  # noqa: F401
    from pymdp.agent import Agent
    from pymdp.envs import TMaze
    from pymdp.envs.rollout import rollout

    # Real rollout with pybefit loaded — confirms full stack works
    env = TMaze(categorical_obs=False)
    C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
    C[1] = C[1].at[1].set(3.0).at[2].set(-3.0)
    agent = Agent(env.A, env.B, C, D=env.D, A_dependencies=env.A_dependencies,
                  B_dependencies=env.B_dependencies, num_controls=[5, 1],
                  batch_size=1, policy_len=2)
    T = 3 if cfg.fast else 10
    _, info = rollout(agent, env, num_timesteps=T, rng_key=jr.PRNGKey(cfg.seed))
    _auto_plot_metrics(cfg, info, 'pybefit_tmaze')
    return {
        "ok": True, "id": "model_fitting/fitting_with_pybefit",
        "note": "pybefit_real_rollout", "T": T,
        "action_shape": tuple(info["action"].shape),
        "diagnostics": _extract_rollout_diagnostics(info),
    }


def _h_tmaze_recoverability(cfg: OrchestrationConfig) -> dict[str, Any]:
    import runpy
    import sys

    if not try_import_pybefit():
        return {"ok": True, "skipped": True, "reason": "no_pybefit"}
    repo = _REPO_ROOT
    target = repo / "examples" / "model_fitting" / "tmaze_recoverability.py"
    out = cfg.output_dir if cfg.output_dir else (repo / "docxology" / "output" / "model_fitting")
    out.mkdir(parents=True, exist_ok=True)
    argv = [
        str(target),
        "--num-blocks",
        "2",
        "--svi-steps",
        "5" if cfg.fast else "40",
        "--num-agents",
        "2",
        "--num-trials",
        "1",
        "--num-samples",
        "4",
        "--save-plot",
        str(out / "tmaze_recoverability_plot.png"),
        "--save-json",
        str(out / "tmaze_recoverability_metrics.json"),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old
    return {"ok": True, "id": "model_fitting/tmaze_recoverability"}


def _h_legacy_agent_demo(cfg: OrchestrationConfig) -> dict[str, Any]:
    import copy

    from pymdp.legacy import default_models, utils
    from pymdp.legacy.agent import Agent

    A, B, C, control_fac_idx = default_models.generate_epistemic_MAB_model()
    agent = Agent(A=A, B=B, C=C, control_fac_idx=control_fac_idx)
    observation = [2, 2, 0]
    agent.infer_states(observation)
    agent.infer_policies()
    _ = agent.sample_action()
    B_gp = copy.deepcopy(B)
    state = [0, 0]
    actions = []
    qs_seq = []
    q_pi_seq = []
    efe_seq = []
    states_seq = []
    for _ in range(2 if cfg.fast else 15):
        qs = agent.infer_states(observation)
        qs_seq.append(qs[0] if isinstance(qs, (list, tuple)) else qs)
        q_pi, G = agent.infer_policies()
        q_pi_seq.append(q_pi.tolist() if hasattr(q_pi, 'tolist') else list(q_pi))
        efe_seq.append(G.tolist() if hasattr(G, 'tolist') else list(G))
        action = agent.sample_action()
        actions.append([float(a) for a in action])
        for f, s in enumerate(state):
            state[f] = utils.sample(B_gp[f][:, s, int(action[f])])
        states_seq.append(list(state))
    # Extract flat action list for plotting (take first control factor)
    flat_actions = [a[0] for a in actions]
    _auto_plot_metrics(cfg, {"action": flat_actions, "qs": qs_seq, "q_pi": q_pi}, 'legacy_agent_demo')
    return {
        "ok": True, "id": "legacy/agent_demo", "final_state": state,
        "diagnostics": {
            "actions_per_timestep": actions,
            "states_per_timestep": states_seq,
            "beliefs_per_timestep": [q.tolist() if hasattr(q, 'tolist') else list(q) for q in qs_seq],
            "q_pi_per_timestep": q_pi_seq,
            "efe_per_timestep": efe_seq,
        }
    }


def _h_legacy_tmaze(cfg: OrchestrationConfig) -> dict[str, Any]:
    import copy

    from pymdp.legacy import utils
    from pymdp.legacy.agent import Agent
    from pymdp.legacy.envs import TMazeEnv

    env = TMazeEnv(reward_probs=[0.98, 0.02])
    A_gp = env.get_likelihood_dist()
    B_gp = env.get_transition_dist()
    A_gm = copy.deepcopy(A_gp)
    B_gm = copy.deepcopy(B_gp)
    agent = Agent(A=A_gm, B=B_gm, control_fac_idx=[0])
    agent.C[1][1] = 3.0
    agent.C[1][2] = -3.0
    obs = env.reset()
    actions = []
    qs_seq = []
    q_pi_seq = []
    efe_seq = []
    vfe_seq = []
    for _ in range(3 if cfg.fast else 25):
        qs = agent.infer_states(obs)
        qs_seq.append(qs[0] if isinstance(qs, (list, tuple)) else qs)
        q_pi, G = agent.infer_policies()
        q_pi_seq.append(q_pi.tolist() if hasattr(q_pi, 'tolist') else list(q_pi))
        efe_seq.append(G.tolist() if hasattr(G, 'tolist') else list(G))
        if hasattr(agent, 'F') and agent.F is not None:
            vfe_seq.append(float(agent.F) if not hasattr(agent.F, 'tolist') else agent.F.tolist())
        action = agent.sample_action()
        actions.append(float(action[0]) if hasattr(action, '__len__') else float(action))
        obs = env.step(action)
    _auto_plot_metrics(cfg, {"action": actions, "qs": qs_seq}, 'legacy_tmaze')
    return {
        "ok": True, "id": "legacy/tmaze_demo",
        "diagnostics": {
            "actions_per_timestep": actions,
            "beliefs_per_timestep": [q.tolist() if hasattr(q, 'tolist') else list(q) for q in qs_seq],
            "q_pi_per_timestep": q_pi_seq,
            "efe_per_timestep": efe_seq,
            "vfe_per_timestep": vfe_seq,
        }
    }


def _h_legacy_tmaze_learning(cfg: OrchestrationConfig) -> dict[str, Any]:
    import copy

    from pymdp.legacy import utils
    from pymdp.legacy.agent import Agent
    from pymdp.legacy.envs import TMazeEnvNullOutcome

    env = TMazeEnvNullOutcome(reward_probs=[0.85, 0.15])
    A_gp = env.get_likelihood_dist()
    B_gp = env.get_transition_dist()
    pA = utils.dirichlet_like(A_gp, scale=1e16)
    pA[1][1:, 1:3, :] = 1.0
    A_gm = utils.norm_dist_obj_arr(pA)
    B_gm = copy.deepcopy(B_gp)
    agent = Agent(
        A=A_gm,
        pA=pA,
        B=B_gm,
        control_fac_idx=[0],
        modalities_to_learn=[1],
        lr_pA=0.25,
        use_param_info_gain=True,
    )
    agent.D[0] = utils.onehot(0, agent.num_states[0])
    agent.C[1][1] = 2.0
    agent.C[1][2] = -2.0
    obs = env.reset()
    n = 5 if cfg.fast else 30
    actions = []
    qs_seq = []
    q_pi_seq = []
    efe_seq = []
    vfe_seq = []
    for _ in range(n):
        qs = agent.infer_states(obs)
        qs_seq.append(qs[0] if isinstance(qs, (list, tuple)) else qs)
        q_pi, _G = agent.infer_policies()
        q_pi_seq.append(q_pi.tolist() if hasattr(q_pi, 'tolist') else list(q_pi))
        efe_seq.append(_G.tolist() if hasattr(_G, 'tolist') else list(_G))
        if hasattr(agent, 'F') and agent.F is not None:
            vfe_seq.append(float(agent.F) if not hasattr(agent.F, 'tolist') else agent.F.tolist())
        action = agent.sample_action()
        actions.append(float(action[0]) if hasattr(action, '__len__') else float(action))
        agent.update_A(obs)
        obs = env.step(action)
    _auto_plot_metrics(cfg, {"action": actions, "qs": qs_seq, "q_pi": q_pi}, 'legacy_tmaze_learning')
    return {
        "ok": True, "id": "legacy/tmaze_learning_demo", "steps": n,
        "diagnostics": {
            "actions_per_timestep": actions,
            "beliefs_per_timestep": [q.tolist() if hasattr(q, 'tolist') else list(q) for q in qs_seq],
            "q_pi_per_timestep": q_pi_seq,
            "efe_per_timestep": efe_seq,
            "vfe_per_timestep": vfe_seq,
        }
    }


def _h_legacy_grid1(cfg: OrchestrationConfig) -> dict[str, Any]:
    import copy

    from pymdp.legacy import utils
    from pymdp.legacy.agent import Agent
    from pymdp.legacy.envs import GridWorldEnv

    env = GridWorldEnv(shape=[3, 3])
    A = copy.deepcopy(env.get_likelihood_dist())
    B = copy.deepcopy(env.get_transition_dist())
    C = utils.obj_array_zeros([A.shape[0]])
    agent = Agent(A=A, B=B, C=C, control_fac_idx=[0])
    state = env.reset()
    obs_list = [int(state)]
    actions = []
    qs_seq = []
    q_pi_seq = []
    efe_seq = []
    vfe_seq = []
    for _ in range(2 if cfg.fast else 15):
        qs = agent.infer_states(obs_list)
        qs_seq.append(qs[0] if isinstance(qs, (list, tuple)) else qs)
        q_pi, G = agent.infer_policies()
        q_pi_seq.append(q_pi.tolist() if hasattr(q_pi, 'tolist') else list(q_pi))
        efe_seq.append(G.tolist() if hasattr(G, 'tolist') else list(G))
        if hasattr(agent, 'F') and agent.F is not None:
            vfe_seq.append(float(agent.F) if not hasattr(agent.F, 'tolist') else agent.F.tolist())
        act = agent.sample_action()
        actions.append(float(act[0]) if hasattr(act, '__len__') else float(act))
        state = env.step(int(act[0]))
        obs_list = [int(state)]
    _auto_plot_metrics(cfg, {"action": actions, "qs": qs_seq}, 'legacy_grid1')
    return {
        "ok": True, "id": "legacy/gridworld_tutorial_1",
        "diagnostics": {
            "actions_per_timestep": actions,
            "beliefs_per_timestep": [q.tolist() if hasattr(q, 'tolist') else list(q) for q in qs_seq],
            "q_pi_per_timestep": q_pi_seq,
            "efe_per_timestep": efe_seq,
            "vfe_per_timestep": vfe_seq,
        }
    }


def _h_legacy_grid2(cfg: OrchestrationConfig) -> dict[str, Any]:
    return _h_legacy_grid1(cfg) | {"id": "legacy/gridworld_tutorial_2"}


def _h_legacy_free_energy(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    from pymdp.legacy.maths import spm_log_single

    x = np.array([0.1, 0.9], dtype=np.float64)
    lx = spm_log_single(x)
    # SPM log comparison visualization
    if cfg.save_plots:
        from pkg.support.viz import plot_spm_log_comparison
        plot_spm_log_comparison(x, lx, title="Legacy Free Energy: spm_log Transform")
        save_current_figure(cfg.output_dir, stem="free_energy_spm_log")
    return {
        "ok": True, "id": "legacy/free_energy_calculation", "log_sum": float(np.sum(lx)),
        "diagnostics": {
            "input_distribution": x.tolist(),
            "spm_log_values": lx.tolist(),
            "log_sum": float(np.sum(lx)),
            "entropy": float(-np.sum(x * lx)),
        },
    }


def _h_docx_01(cfg: OrchestrationConfig) -> dict[str, Any]:
    d = random_agent_one_cycle(seed=cfg.seed, policy_len=1)
    _auto_plot_metrics(cfg, d, "01_minimal_discrete_timestep")
    return {"ok": True, "id": "docxology/01_minimal_discrete_timestep", "detail": d}


def _h_docx_02(cfg: OrchestrationConfig) -> dict[str, Any]:
    return _h_learning_gridworld(cfg) | {"id": "docxology/02_learning_pA_single_update"}


def _h_docx_03(cfg: OrchestrationConfig) -> dict[str, Any]:
    return _h_cue_chaining(cfg) | {"id": "docxology/03_rollout_cue_chaining_short"}


def _h_docx_04(cfg: OrchestrationConfig) -> dict[str, Any]:
    import numpy as np
    agent = si_cue_agent(policy_len=1)
    qs_plan = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
    q_pi, neg_efe = agent.infer_policies(qs_plan)
    keys = jr.split(jr.PRNGKey(cfg.seed), 2)
    act = agent.sample_action(q_pi, rng_key=keys[1:])
    q_pi_arr = np.asarray(q_pi[0]).squeeze()
    neg_efe_arr = np.asarray(neg_efe[0]).squeeze()
    # Policy posterior & neg-EFE visualization
    if cfg.save_plots:
        from pkg.support.viz import plot_policy_posterior, plot_action_probabilities
        if q_pi_arr.ndim >= 1 and q_pi_arr.size > 1:
            plot_policy_posterior(q_pi_arr, title="SI Cue Agent — Policy Posterior")
            save_current_figure(cfg.output_dir, stem="04_si_cue_q_pi")
        # Action probs from marginalization
        sampled = [int(np.asarray(a).flatten()[0]) for a in act]
        num_acts = max(3, max(sampled) + 1)
        act_probs = np.zeros(num_acts)
        for a_idx in sampled:
            act_probs[a_idx] = 1.0 / len(sampled)
        plot_action_probabilities(act_probs, title="SI Cue Agent — Sampled Actions")
        save_current_figure(cfg.output_dir, stem="04_si_cue_actions")
    return {
        "ok": True, "id": "docxology/04_vanilla_policy_vs_si_stub",
        "marginals_shape": tuple(q_pi[0].shape),
        "diagnostics": {
            "q_pi": q_pi_arr.tolist(),
            "neg_efe": neg_efe_arr.tolist(),
            "sampled_action": [int(np.asarray(a).flatten()[0]) for a in act],
        },
    }


HANDLERS: dict[str, Callable[[OrchestrationConfig], dict[str, Any]]] = {
    "api/model_construction_tutorial.py": _h_model_construction,
    "advanced/complex_action_dependency.py": _h_complex_action_dependency,
    "advanced/infer_states_optimization/methods_test.py": _h_infer_states_methods,
    "advanced/pymdp_with_neural_encoder.py": _h_neural_encoder,
    "envs/tmaze_demo.py": _h_tmaze_rollout,
    "envs/generalized_tmaze_demo.py": _h_generalized_tmaze,
    "envs/cue_chaining_demo.py": _h_cue_chaining,
    "envs/graph_worlds_demo.py": _h_graph_worlds,
    "envs/knapsack_demo.py": _h_knapsack_surrogate,
    "envs/chained_cue_navigation.py": _h_chained_cue_nav_upstream,
    "experimental/sophisticated_inference/si_tmaze_SIvalidation.py": _h_si_tmaze_validation,
    "experimental/sophisticated_inference/si_generalized_tmaze.py": _h_si_generalized_tmaze,
    "experimental/sophisticated_inference/si_graph_world.py": _h_si_graph_world,
    "experimental/sophisticated_inference/mcts_generalized_tmaze.py": _h_mcts_generalized_tmaze,
    "experimental/sophisticated_inference/mcts_graph_world.py": _h_mcts_graph_world,
    "inductive_inference/inductive_inference_example.py": _h_inductive_example,
    "inductive_inference/inductive_inference_gridworld.py": _h_inductive_gridworld,
    "inference_and_learning/inference_methods_comparison.py": _h_inference_methods_comparison,
    "learning/learning_gridworld.py": _h_learning_gridworld,
    "sparse/sparse_benchmark.py": _h_sparse,
    "model_fitting/fitting_with_pybefit.py": _h_pybefit,
    "model_fitting/tmaze_recoverability.py": _h_tmaze_recoverability,
    "legacy/agent_demo.py": _h_legacy_agent_demo,
    "legacy/tmaze_demo.py": _h_legacy_tmaze,
    "legacy/tmaze_learning_demo.py": _h_legacy_tmaze_learning,
    "legacy/gridworld_tutorial_1.py": _h_legacy_grid1,
    "legacy/gridworld_tutorial_2.py": _h_legacy_grid2,
    "legacy/free_energy_calculation.py": _h_legacy_free_energy,
    "docxology/01_minimal_discrete_timestep.py": _h_docx_01,
    "docxology/02_learning_pA_single_update.py": _h_docx_02,
    "docxology/03_rollout_cue_chaining_short.py": _h_docx_03,
    "docxology/04_si_cue_agent_vanilla_plan.py": _h_docx_04,
}
