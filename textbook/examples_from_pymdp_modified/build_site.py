#!/usr/bin/env python3
import html
import json
from pathlib import Path


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>pymdp Modified Examples - Artifacts</title>
  <base href=".." />
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; }
    .layout { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }
    .sidebar { background:#111; color:#eee; overflow:auto; }
    .sidebar h1 { font-size: 16px; padding: 12px 16px; margin: 0; border-bottom: 1px solid #222; }
    .sidebar a { display:block; padding: 8px 16px; color:#ddd; text-decoration:none; }
    .sidebar a:hover { background:#1e1e1e; }
    .content { overflow:auto; padding: 16px; }
    h2 { margin-top: 0; }
    .card { border:1px solid #ddd; border-radius:6px; padding: 12px; margin: 12px 0; }
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
    .muted { color:#666; font-size: 13px; }
    code, pre { background:#f8f8f8; padding:4px 6px; border-radius:4px; }
    img { max-width: 100%; height: auto; border:1px solid #ddd; border-radius:4px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border:1px solid #ddd; padding:6px 8px; }
    th { background:#fafafa; }
  </style>
  <script>
    function show(id){
      document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
      document.getElementById(id).style.display = 'block';
    }
    window.addEventListener('DOMContentLoaded', () => { const first = document.querySelector('.section'); if (first) first.style.display='block'; });
  </script>
  </head>
  <body>
    <div class="layout">
      <div class="sidebar">
        <h1>Artifacts</h1>
        {sidebar}
      </div>
      <div class="content">
        {sections}
      </div>
    </div>
  </body>
</html>
"""


def render_example(name: str, ex_dir: Path) -> str:
    files = sorted([p.name for p in ex_dir.glob('*') if p.is_file()])
    imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg'))]
    csvs = [f for f in files if f.lower().endswith('.csv')]
    jsons = [f for f in files if f.lower().endswith('.json')]
    logs = [f for f in files if f.lower().endswith('.log')]

    blocks = []
    if imgs:
        blocks.append('<div class="card"><h3>Images</h3><div class="grid">' + ''.join(
            f'<div><div class="muted">{html.escape(img)}</div><a href="{name}/{html.escape(img)}" target="_blank"><img src="{name}/{html.escape(img)}" /></a></div>'
            for img in imgs
        ) + '</div></div>')
    if csvs:
        blocks.append('<div class="card"><h3>CSV</h3><ul>' + ''.join(
            f'<li><a href="{name}/{html.escape(csv)}" target="_blank">{html.escape(csv)}</a></li>' for csv in csvs
        ) + '</ul></div>')
    if jsons:
        blocks.append('<div class="card"><h3>JSON</h3><ul>' + ''.join(
            f'<li><a href="{name}/{html.escape(js)}" target="_blank">{html.escape(js)}</a></li>' for js in jsons
        ) + '</ul></div>')
    if logs:
        blocks.append('<div class="card"><h3>Logs</h3><ul>' + ''.join(
            f'<li><a href="{name}/{html.escape(l)}" target="_blank">{html.escape(l)}</a></li>' for l in logs
        ) + '</ul></div>')

    if not blocks:
        blocks.append('<div class="muted">No artifacts found for this example.</div>')

    return f'<div id="ex-{html.escape(name)}" class="section" style="display:none"><h2>{html.escape(name)}</h2>' + ''.join(blocks) + '</div>'


def main():
    root = Path(__file__).resolve().parent
    outputs = root / 'outputs'
    site = outputs / 'site'
    site.mkdir(parents=True, exist_ok=True)

    example_dirs = sorted([p for p in outputs.iterdir() if p.is_dir() and p.name != 'site'])
    sidebar = ''.join(
        f'<a href="#" onclick="show(\'ex-{html.escape(p.name)}\');return false;">{html.escape(p.name)}</a>'
        for p in example_dirs
    )
    sections = ''.join(render_example(p.name, p) for p in example_dirs)

    index = TEMPLATE.replace('{sidebar}', sidebar).replace('{sections}', sections)
    (site / 'index.html').write_text(index, encoding='utf-8')
    print(f"Wrote {site / 'index.html'}")


if __name__ == '__main__':
    main()


