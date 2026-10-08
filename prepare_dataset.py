import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import maximum_filter
from skimage.feature import local_binary_pattern
from scipy.spatial.transform import Rotation

VIEWS = [(0, 0, 0), (0, 180, 0), (0, 90, 0), (0, -90, 0), (90, 0, 0), (-90, 0, 0)]
DEFAULT_SIZE = 128
DEFAULT_POINTS = 20000
DEFAULT_CLASSES = ("chair", "table")
CACHE_VERSION = 2


def find_root(root):
    root = Path(root)
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for candidate in candidates:
        if (candidate / "chair" / "train").exists():
            return candidate
    raise FileNotFoundError(f"Dossier ModelNet10 introuvable dans {root}")


def load_points(path, n_points):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError("maillage vide")
    points = np.asarray(mesh.sample(n_points), dtype=np.float32)
    points -= points.mean(axis=0)
    if not np.isfinite(points).all():
        raise ValueError("le maillage contient des points non finis")
    scale = np.linalg.norm(points, axis=1).max()
    if scale <= 0:
        raise ValueError("objet de taille nulle")
    return points / scale


def depth_view(points, euler, size):
    rotated = Rotation.from_euler("xyz", euler, degrees=True).apply(points)
    ix = np.clip(((rotated[:, 0] + 1) * 0.5 * (size - 1)).astype(int), 0, size - 1)
    iy = np.clip(((1 - (rotated[:, 1] + 1) * 0.5) * (size - 1)).astype(int), 0, size - 1)
    depth = np.zeros((size, size), dtype=np.float32)
    np.maximum.at(depth, (iy, ix), (rotated[:, 2] + 1) * 0.5 + 0.01)
    depth = maximum_filter(depth, size=2)
    return np.clip(depth * 255, 0, 255).astype(np.uint8)


def lbp_image(image):
    values = local_binary_pattern(image, P=8, R=1, method="uniform")
    maximum = values.max()
    if maximum <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip(values / maximum * 255, 0, 255).astype(np.uint8)


def make_grid(images):
    return np.vstack((np.hstack(images[:3]), np.hstack(images[3:])))


def process_job(job):
    off_path, output_root, split, class_name, size, n_points, cache_valid = job
    off_path = Path(off_path)
    output_root = Path(output_root)
    raw_path = output_root / "raw" / split / class_name / f"{off_path.stem}.png"
    lbp_path = output_root / "lbp" / split / class_name / f"{off_path.stem}.png"
    if cache_valid and raw_path.exists() and lbp_path.exists():
        return class_name, split, "cached"
    try:
        points = load_points(off_path, n_points)
        views = [depth_view(points, angles, size) for angles in VIEWS]
        raw_grid = make_grid(views)
        lbp_grid = make_grid([lbp_image(view) for view in views])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        lbp_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raw_grid, mode="L").save(raw_path)
        Image.fromarray(lbp_grid, mode="L").save(lbp_path)
        return class_name, split, "created"
    except Exception as error:
        return class_name, split, f"error: {off_path}: {error}"


def read_cache_config(path, expected):
    path = Path(path)
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")) == expected


def main():
    parser = argparse.ArgumentParser(description="Convertit ModelNet10 OFF en mosaïques 6 vues avec et sans LBP")
    parser.add_argument("--root", required=True, help="Dossier contenant ModelNet10")
    parser.add_argument("--out", default="dataset", help="Dossier de sortie")
    parser.add_argument("--classes", nargs=2, default=DEFAULT_CLASSES, metavar=("CLASSE1", "CLASSE2"),
                        help="Deux classes à convertir (défaut : chair table)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Taille de chaque vue")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="Points échantillonnés par objet")
    parser.add_argument("--workers", type=int, default=2, help="Nombre de processus")
    args = parser.parse_args()

    if len(set(args.classes)) != 2:
        parser.error("les deux classes doivent être différentes")
    if args.size <= 0 or args.points <= 0 or args.workers <= 0:
        parser.error("--size, --points et --workers doivent être strictement positifs")

    root = find_root(args.root)
    output_root = Path(args.out).resolve()
    jobs = []
    available_classes = {path.name for path in root.iterdir() if path.is_dir()}
    missing_classes = sorted(set(args.classes) - available_classes)
    if missing_classes:
        parser.error(f"classes introuvables dans {root} : {', '.join(missing_classes)}")

    cache_configs = {}
    for class_name in args.classes:
        for split in ("train", "test"):
            split_dir = root / class_name / split
            if not split_dir.is_dir():
                parser.error(f"partition introuvable : {split_dir}")
            off_paths = sorted(split_dir.glob("*.off"))
            if not off_paths:
                parser.error(f"aucun fichier OFF dans {split_dir}")
            expected_config = {"version": CACHE_VERSION, "size": args.size, "points": args.points}
            cache_path = output_root / ".cache" / class_name / f"{split}.json"
            cache_valid = read_cache_config(cache_path, expected_config)
            cache_configs[(class_name, split)] = (cache_path, expected_config)
            jobs.extend(
                (str(off_path), str(output_root), split, class_name, args.size, args.points, cache_valid)
                for off_path in off_paths
            )

    print(f"Racine ModelNet10 : {root}")
    print(f"Classes sélectionnées : {', '.join(args.classes)}")
    print(f"Objets à convertir : {len(jobs)}")
    print(f"Sortie : {output_root}")
    created = cached = errors = 0
    failed_groups = set()
    with Pool(processes=args.workers) as pool:
        for index, result in enumerate(pool.imap_unordered(process_job, jobs), 1):
            class_name, split, status = result
            if status == "created":
                created += 1
            elif status == "cached":
                cached += 1
            else:
                errors += 1
                failed_groups.add((class_name, split))
                print(f"[{status}]")
            if index % 200 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} | créés={created} | déjà présents={cached} | erreurs={errors}")

    for group, (cache_path, config) in cache_configs.items():
        if group not in failed_groups:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Terminé.")
    print(f"Classes converties : {', '.join(args.classes)}")
    print(f"Erreurs de conversion : {errors}")
    print(f"Dataset raw : {output_root / 'raw'}")
    print(f"Dataset LBP : {output_root / 'lbp'}")
    if errors:
        raise RuntimeError(f"{errors} objet(s) n'ont pas pu être convertis.")


if __name__ == "__main__":
    main()
