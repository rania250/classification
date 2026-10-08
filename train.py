import argparse
import html
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEFAULT_CLASSES = ("chair", "table")
VALIDATION_FRACTION = 0.15


class ViewGridCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(data_dir, batch_size, classes, seed=42):
    image_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_ds = datasets.ImageFolder(str(Path(data_dir) / "train"), image_transform)
    validation_ds = datasets.ImageFolder(str(Path(data_dir) / "train"), image_transform)
    test_ds = datasets.ImageFolder(str(Path(data_dir) / "test"), image_transform)
    class_names = filter_classes(train_ds, classes)
    filter_classes(validation_ds, class_names)
    filter_classes(test_ds, class_names)

    train_indices, validation_indices = stratified_split(train_ds.targets, seed=seed)
    train_dl = DataLoader(
        Subset(train_ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_dl = DataLoader(
        Subset(validation_ds, validation_indices), batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_ds, train_dl, validation_dl, test_dl, class_names


def filter_classes(dataset, class_names):
    if len(class_names) != 2 or len(set(class_names)) != 2:
        raise ValueError("Il faut sélectionner exactement deux classes différentes.")
    missing = sorted(set(class_names) - set(dataset.classes))
    if missing:
        raise ValueError(
            f"Classes absentes de {dataset.root}: {', '.join(missing)}. "
            f"Classes disponibles : {', '.join(dataset.classes)}"
        )

    original_indices = {dataset.class_to_idx[name]: index for index, name in enumerate(class_names)}
    samples = [
        (path, original_indices[target])
        for path, target in dataset.samples
        if target in original_indices
    ]
    counts = [sum(target == index for _, target in samples) for index in range(len(class_names))]
    if any(count == 0 for count in counts):
        raise ValueError(f"Chaque classe doit contenir des images : {dict(zip(class_names, counts))}")

    dataset.classes = list(class_names)
    dataset.class_to_idx = {name: index for index, name in enumerate(class_names)}
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [target for _, target in samples]
    return list(class_names)


def stratified_split(targets, validation_fraction=VALIDATION_FRACTION, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("La proportion de validation doit être comprise entre 0 et 1.")
    generator = torch.Generator().manual_seed(seed)
    train_indices, validation_indices = [], []
    for class_index in sorted(set(targets)):
        indices = torch.tensor(
            [index for index, target in enumerate(targets) if target == class_index],
            dtype=torch.int64,
        )
        if len(indices) < 2:
            raise ValueError("Il faut au moins deux objets d'entraînement par classe.")
        indices = indices[torch.randperm(len(indices), generator=generator)].tolist()
        validation_count = min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    return train_indices, validation_indices


def save_error_gallery(output_dir, paths, true_labels, predicted_labels, class_names):
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for index, (path, true_label, predicted_label) in enumerate(zip(paths, true_labels, predicted_labels)):
        if true_label == predicted_label:
            continue
        source = Path(path)
        name = f"{index:05d}_true-{class_names[true_label]}_pred-{class_names[predicted_label]}_{source.name}"
        shutil.copy2(source, image_dir / name)
        errors.append((name, class_names[true_label], class_names[predicted_label], str(source)))
    (output_dir / "errors.json").write_text(json.dumps([
        {"image": n, "true": t, "predicted": p, "path": path} for n, t, p, path in errors
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "report.html").open("w", encoding="utf-8") as page:
        page.write("<html><head><meta charset='utf-8'><title>Galerie des erreurs</title>")
        page.write("<style>body{font-family:Arial}.grid{display:flex;flex-wrap:wrap;gap:12px}.item{width:260px;border:1px solid #ccc;padding:8px}img{width:250px;height:167px;object-fit:contain;background:#eee}.path{font-size:11px;word-break:break-all}</style></head><body>")
        page.write(f"<h1>Images mal classées : {len(errors)}</h1><div class='grid'>")
        for name, true_name, predicted_name, path in errors:
            page.write(f"<div class='item'><img src='images/{html.escape(name)}'><br>Vrai : {html.escape(true_name)}<br>Prédit : {html.escape(predicted_name)}<br><span class='path'>{html.escape(path)}</span></div>")
        page.write("</div></body></html>")
    return len(errors)


def classification_metrics(true_labels, predicted_labels, class_names):
    labels = list(range(len(class_names)))
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (true_labels, predicted_labels), 1)

    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    precision = np.divide(
        matrix.diagonal(), predicted_count, out=np.zeros(len(labels), dtype=float), where=predicted_count != 0
    )
    recall = np.divide(
        matrix.diagonal(), support, out=np.zeros(len(labels), dtype=float), where=support != 0
    )
    f1_score = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(labels), dtype=float),
        where=(precision + recall) != 0,
    )
    total = int(support.sum())
    accuracy = float(matrix.diagonal().sum() / total) if total else 0.0

    report = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1-score": float(f1_score[index]),
            "support": float(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    report["accuracy"] = accuracy
    for average_name, weights in (
        ("macro avg", np.ones(len(labels), dtype=float)),
        ("weighted avg", support.astype(float)),
    ):
        denominator = weights.sum()
        report[average_name] = {
            metric_name: float(np.average(values, weights=weights)) if denominator else 0.0
            for metric_name, values in (("precision", precision), ("recall", recall), ("f1-score", f1_score))
        }
        report[average_name]["support"] = float(total)
    return report, matrix


def evaluate(model, loader, dataset, device, class_names, output_dir):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            all_pred.extend(predictions)
            all_true.extend(labels.tolist())
    report, matrix = classification_metrics(all_true, all_pred, class_names)
    print(f"Accuracy test : {report['accuracy'] * 100:.2f}%")
    print(f"{'':16}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for class_name in class_names:
        metrics = report[class_name]
        print(
            f"{class_name:16}{metrics['precision']:>10.2f}{metrics['recall']:>10.2f}"
            f"{metrics['f1-score']:>10.2f}{metrics['support']:>10.0f}"
        )
    for average_name in ("accuracy", "macro avg", "weighted avg"):
        if average_name == "accuracy":
            print(f"{average_name:16}{'':>10}{'':>10}{report[average_name]:>10.2f}{len(all_true):>10}")
        else:
            metrics = report[average_name]
            print(
                f"{average_name:16}{metrics['precision']:>10.2f}{metrics['recall']:>10.2f}"
                f"{metrics['f1-score']:>10.2f}{metrics['support']:>10.0f}"
            )
    print("Matrice de confusion :")
    print(matrix)
    errors = save_error_gallery(output_dir, [path for path, _ in dataset.samples], all_true, all_pred, class_names)
    print(f"Images mal classées : {errors}")
    print(f"Galerie : {Path(output_dir) / 'report.html'}")
    return report


def train_one(data_dir, output_root, epochs, batch_size, lr, device, classes):
    test_ds, train_dl, validation_dl, test_dl, class_names = make_loaders(
        data_dir, batch_size, classes
    )
    print(f"Classes : {class_names}")
    print(f"Train : {len(train_dl.dataset)} | Validation : {len(validation_dl.dataset)} | Test : {len(test_ds)}")
    model = ViewGridCNN(len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        scheduler.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in validation_dl:
                predictions = model(images.to(device)).argmax(1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
        accuracy = correct / total
        print(f"Epoch {epoch:02d}/{epochs} | loss={total_loss / len(train_dl.dataset):.4f} | val_acc={accuracy * 100:.2f}%")
        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(output_root) / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    report = evaluate(model, test_dl, test_ds, device, class_names, Path(output_root) / "errors")
    (Path(output_root) / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Meilleure accuracy validation : {best_validation_accuracy * 100:.2f}%")
    print(f"Modèle : {model_path}")


def main():
    parser = argparse.ArgumentParser(description="CNN ModelNet10 sur mosaïques 2x3")
    parser.add_argument("--data", required=True, help="dataset/raw ou dataset/lbp")
    parser.add_argument("--classes", nargs=2, default=DEFAULT_CLASSES, metavar=("CLASSE1", "CLASSE2"),
                        help="Deux classes à classifier (défaut : chair table)")
    parser.add_argument("--out", default=None, help="Dossier de sortie des résultats")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if len(set(args.classes)) != 2:
        parser.error("les deux classes doivent être différentes")
    if args.epochs <= 0 or args.bs <= 0 or args.lr <= 0:
        parser.error("--epochs, --bs et --lr doivent être strictement positifs")
    seed_everything()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA a été demandé mais n'est pas disponible")
    output_root = Path(args.out) if args.out else Path(args.data).parent / (Path(args.data).name + "_results")
    print(f"Device : {device}")
    train_one(args.data, output_root, args.epochs, args.bs, args.lr, device, args.classes)


if __name__ == "__main__":
    main()
