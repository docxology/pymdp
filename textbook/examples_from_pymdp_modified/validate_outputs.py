#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

# Use non-interactive backend for plotting
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use(os.environ["MPLBACKEND"], force=True)
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_tmaze_timeline(log_text: str):
    rows = []
    step_re = re.compile(r"\[Step\s+(\d+)\]\s+Observation:\s*\[(.*?),\s*(.*?),\s*(.*?)\]")
    action_re = re.compile(r"\[Step\s+(\d+)\]\s+Action:\s*\[(.*?)\]")
    # Capture latest action for each step prior to observation line
    last_action = {}
    for line in log_text.splitlines():
        m_a = action_re.search(line)
        if m_a:
            t = int(m_a.group(1))
            last_action[t] = m_a.group(2)
            continue
        m = step_re.search(line)
        if m:
            t = int(m.group(1))
            loc, rew, cue = m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
            action = last_action.get(t, "")
            rows.append([t, loc, rew, cue, action])
    return rows


def parse_agent_demo(log_text: str):
    # Expect lines like:
    # 0: Observation state_observation: 2
    # 0: Observation reward: 2
    # 0: Observation decision_proprioceptive: 0
    # 0: Beliefs about reward_level: [0.5 0.5]
    # 0: Beliefs about decision_state: [1.0e+00 1.0e-32 1.2e-16]
    # 1
    # 0: Action: [0. 2.] / State: [0, 2]
    rows = []
    t, obs = None, {}
    belief = {}
    action = None
    for line in log_text.splitlines():
        m_obs = re.match(r"(\d+):\s+Observation\s+(\w+):\s*(\d+)", line)
        if m_obs:
            t = int(m_obs.group(1))
            obs_key, obs_val = m_obs.group(2), m_obs.group(3)
            obs.setdefault(t, {})[obs_key] = obs_val
            continue
        m_b = re.match(r"(\d+):\s+Beliefs about\s+([^:]+):\s*\[(.*?)\]", line)
        if m_b:
            t = int(m_b.group(1))
            k = m_b.group(2).strip()
            vals = [v for v in re.split(r"\s+", m_b.group(3).strip()) if v]
            belief.setdefault(t, {})[k] = vals
            continue
        m_a = re.match(r"(\d+):\s+Action:\s*\[(.*?)\]\s*/\s*State:\s*\[(.*?)\]", line)
        if m_a:
            t = int(m_a.group(1))
            action = m_a.group(2)
            state = m_a.group(3)
            ob = obs.get(t, {})
            bel = belief.get(t, {})
            rows.append([
                t,
                ob.get("state_observation", ""),
                ob.get("reward", ""),
                ob.get("decision_proprioceptive", ""),
                action,
                state,
                json.dumps(bel),
            ])
            action = None
            continue
    return rows

def parse_gridworld_qs(log_text: str):
    # Lines printed like: [0.111 0.222 ...]
    rows = []
    t = 0
    vec_re = re.compile(r"^\[(?:\s*[-+eE0-9\.]+\s*)+\]$")
    for line in log_text.splitlines():
        s = line.strip()
        if vec_re.match(s):
            # normalize whitespace and split
            vals = re.split(r"\s+", s.strip("[] "))
            rows.append([t] + vals)
            t += 1
    return rows

def parse_free_energy_metrics(log_text: str):
    # Look for lines like "Initial F = 0.916" and "Final F = 0.693"
    initial = re.search(r"Initial\s+F\s*=\s*([-+eE0-9\.]+)", log_text)
    final = re.search(r"Final\s+F\s*=\s*([-+eE0-9\.]+)", log_text)
    surprise = re.search(r"Surprise:\s*([-+eE0-9\.]+)", log_text)
    out = {}
    if initial:
        out["initial_F"] = float(initial.group(1))
    if final:
        out["final_F"] = float(final.group(1))
    if surprise:
        out["surprise"] = float(surprise.group(1))
    return out

def parse_grid_positions(log_text: str):
    # Lines like: Grid position for agent 2 at time 0: (3, 3)
    rows = []
    pos_re = re.compile(r"Grid position.*time\s+(\d+):\s*\((\d+),\s*(\d+)\)")
    for line in log_text.splitlines():
        m = pos_re.search(line)
        if m:
            rows.append([int(m.group(1)), int(m.group(2)), int(m.group(3))])
    return rows


def plot_tmaze_timeline(csv_path: Path):
    if plt is None:
        return None
    xs, rewards, actions = [], [], []
    action_to_idx = {}
    next_idx = 0
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            t, loc, rew, cue, action = [s.strip() for s in line.rstrip().split(",", 4)]
            xs.append(int(t))
            rewards.append(1 if "Reward" in rew else (0 if "No reward" in rew else -1))
            if action not in action_to_idx:
                action_to_idx[action] = next_idx
                next_idx += 1
            actions.append(action_to_idx[action])
    fig = plt.figure(figsize=(10, 4))
    plt.plot(xs, rewards, label="reward (+1/0/-1)")
    plt.step(xs, actions, where="post", label="action (cat)")
    plt.xlabel("t")
    plt.legend()
    out = csv_path.parent / "timeline_plot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def plot_agent_demo_timeline(csv_path: Path):
    if plt is None:
        return None
    xs, actions = [], []
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip().split(",")
            if len(parts) < 7:
                continue
            t = int(parts[0])
            action = parts[4]
            xs.append(t)
            actions.append(action)
    # Map actions to category index
    uniq = {a: i for i, a in enumerate(sorted(set(actions)))}
    ys = [uniq[a] for a in actions]
    fig = plt.figure(figsize=(10, 4))
    plt.step(xs, ys, where="post")
    plt.yticks(list(uniq.values()), list(uniq.keys()), fontsize=8)
    plt.xlabel("t")
    plt.title("agent_demo actions")
    out = csv_path.parent / "actions_plot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def plot_grid_path(csv_path: Path, shape=(7, 7)):
    if plt is None:
        return None
    ts, rs, cs = [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            t, r, c = map(int, line.rstrip().split(","))
            ts.append(t); rs.append(r); cs.append(c)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(cs, rs, marker="o")
    plt.gca().invert_yaxis()
    plt.xlim(-0.5, shape[1]-0.5)
    plt.ylim(shape[0]-0.5, -0.5)
    plt.grid(True, linestyle=":")
    plt.title("Grid path")
    out = csv_path.parent / "path_plot.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
    return out.name


def plot_beliefs_series(csv_path: Path):
    if plt is None:
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip().split(",")
        cols = header[1:]
        data = [[float(x) for x in line.rstrip().split(",")[1:]] for line in f]
    if not data:
        return None
    fig = plt.figure(figsize=(10, 4))
    # plot up to first 9 series to avoid clutter
    for i, name in enumerate(cols[:9]):
        plt.plot([row[i] for row in data], label=name)
    plt.xlabel("t")
    plt.ylabel("belief prob")
    plt.legend(fontsize=6, ncol=3)
    out = csv_path.parent / "beliefs_series_plot.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
    return out.name


def main():
    here = Path(__file__).resolve().parent
    outputs = here / "outputs"
    summary = {}
    for ex_dir in sorted(outputs.glob("*")):
        if not ex_dir.is_dir():
            continue
        files = {p.name: p for p in ex_dir.glob("*") if p.is_file()}
        manifest = {
            "example": ex_dir.name,
            "files": sorted(files.keys()),
        }
        # T-Maze: derive timeline.csv if possible
        if ex_dir.name in {"tmaze_demo", "tmaze_learning_demo"} and "run.log" in files:
            log_text = files["run.log"].read_text(encoding="utf-8", errors="ignore")
            rows = parse_tmaze_timeline(log_text)
            if rows:
                out_csv = ex_dir / "timeline.csv"
                if not out_csv.exists():
                    with out_csv.open("w", encoding="utf-8") as f:
                        f.write("t,location,reward,cue,action\n")
                        for r in rows:
                            f.write(",".join(map(str, r)) + "\n")
                    manifest["files"].append("timeline.csv")
            # Produce a simple time-series plot
            plot_name = plot_tmaze_timeline(ex_dir / "timeline.csv")
            if plot_name and plot_name not in manifest["files"]:
                manifest["files"].append(plot_name)
        # agent_demo: derive timeline.csv with beliefs json column
        if ex_dir.name == "agent_demo" and "run.log" in files:
            log_text = files["run.log"].read_text(encoding="utf-8", errors="ignore")
            rows = parse_agent_demo(log_text)
            if rows:
                out_csv = ex_dir / "timeline.csv"
                if not out_csv.exists():
                    with out_csv.open("w", encoding="utf-8") as f:
                        f.write("t,state_obs,reward_obs,decision_obs,action,state,beliefs_json\n")
                        for r in rows:
                            f.write(",".join(map(str, r)) + "\n")
                    manifest["files"].append("timeline.csv")
            plot_name = plot_agent_demo_timeline(ex_dir / "timeline.csv")
            if plot_name and plot_name not in manifest["files"]:
                manifest["files"].append(plot_name)
        # gridworld: parse beliefs sequence from log if present
        if ex_dir.name in {"gridworld_tutorial_1", "gridworld_tutorial_2"} and "run.log" in files:
            log_text = files["run.log"].read_text(encoding="utf-8", errors="ignore")
            rows = parse_gridworld_qs(log_text)
            if rows:
                out_csv = ex_dir / "beliefs_series.csv"
                if not out_csv.exists():
                    with out_csv.open("w", encoding="utf-8") as f:
                        # header: t plus indices 0..N-1 (unknown N, infer from first row)
                        header = ["t"] + [f"q{i}" for i in range(len(rows[0]) - 1)]
                        f.write(",".join(header) + "\n")
                        for r in rows:
                            f.write(",".join(map(str, r)) + "\n")
                    manifest["files"].append("beliefs_series.csv")
            # Plot series if present
            if (ex_dir / "beliefs_series.csv").exists():
                plot_name = plot_beliefs_series(ex_dir / "beliefs_series.csv")
                if plot_name and plot_name not in manifest["files"]:
                    manifest["files"].append(plot_name)
        # free_energy_calculation: extract metrics JSON
        if ex_dir.name == "free_energy_calculation" and "run.log" in files:
            log_text = files["run.log"].read_text(encoding="utf-8", errors="ignore")
            metrics = parse_free_energy_metrics(log_text)
            if metrics:
                (ex_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                manifest.setdefault("files", []).append("metrics.json")
        # inductive_inference_gridworld: grid path CSV
        if ex_dir.name == "inductive_inference_gridworld" and "run.log" in files:
            log_text = files["run.log"].read_text(encoding="utf-8", errors="ignore")
            rows = parse_grid_positions(log_text)
            if rows:
                out_csv = ex_dir / "path.csv"
                if not out_csv.exists():
                    with out_csv.open("w", encoding="utf-8") as f:
                        f.write("t,row,col\n")
                        for r in rows:
                            f.write(",".join(map(str, r)) + "\n")
                    manifest["files"].append("path.csv")
            if (ex_dir / "path.csv").exists():
                plot_name = plot_grid_path(ex_dir / "path.csv")
                if plot_name and plot_name not in manifest["files"]:
                    manifest["files"].append(plot_name)
        (ex_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary[ex_dir.name] = manifest["files"]

    (outputs / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


