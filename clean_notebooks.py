
import json
import sys
from pathlib import Path

def clean_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    modified = False
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        modified = True
    for cell in nb.get("cells", []):
        if "widgets" in cell.get("metadata", {}):
            del cell["metadata"]["widgets"]
            modified = True
    for cell in nb.get("cells", []):
        original = cell.get("outputs", [])
        cleaned = [o for o in original if "application/vnd.jupyter.widget-view+json" not in o.get("data", {})]
        if len(cleaned) != len(original):
            cell["outputs"] = cleaned
            modified = True
    if modified:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"  cleaned → {path.name}")
    else:
        print(f"  ok      → {path.name}")

notebooks = list(Path("/content/Uk-Energy-Modelling/notebooks").glob("*.ipynb"))
print(f"Scanning {len(notebooks)} notebooks...\n")
for nb in notebooks:
    clean_notebook(nb)
print("\nDone.")
