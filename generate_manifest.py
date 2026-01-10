#!/usr/bin/env python3
"""
Auto-generate manifest.json from .glb files in the objects/ directories.
Run this script whenever you add new .glb files to automatically update the manifest.
"""

import json
from pathlib import Path

def generate_manifest():
    objects_dir = Path(__file__).parent / "objects"
    manifest_path = objects_dir / "manifest.json"
    
    manifest = {
        "shirts": [],
        "pants": [],
        "shoes": [],
        "dresses": [],
        "shorts": []
    }
    
    # Scan each category folder
    for category in ["shirts", "pants", "shoes", "dresses", "shorts"]:
        category_dir = objects_dir / category
        if category_dir.exists():
            # Find all .glb files in the category folder
            glb_files = sorted(category_dir.glob("*.glb"))
            manifest[category] = [f.name for f in glb_files]
            print(f"Found {len(glb_files)} files in {category}/")
    
    # Write manifest.json
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest generated at {manifest_path}")
    print(f"  Total items: {sum(len(v) for v in manifest.values())}")
    return manifest

if __name__ == "__main__":
    generate_manifest()
