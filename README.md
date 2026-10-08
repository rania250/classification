# Classification de maillages 3D en deux classes

Ce projet transforme des maillages ModelNet10 (`.off`) en six vues en niveaux de gris, les assemble en une mosaïque 2 × 3 et entraîne un CNN pour distinguer deux classes. Par défaut, les classes sont `chair` et `table` ; elles peuvent être remplacées par n'importe quelles deux catégories présentes dans ModelNet10.

Les fichiers ModelNet10 ne contenant pas de textures ni de photographies, les vues produites sont des projections orthographiques de profondeur, et non des photos réalistes. Chaque maillage est recentré et normalisé pour tenir dans une sphère unité avant la génération des vues avant, arrière, gauche, droite, dessus et dessous. La variante LBP applique un LBP uniforme (8 voisins, rayon 1) séparément à chacune des six vues avant de les assembler.

## Préparation

Dans l'environnement virtuel du projet, installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

Cela installe `numpy`, `trimesh`, `Pillow`, `scipy`, `scikit-image`, `torch` et `torchvision`. Les versions de `torch` et `torchvision` doivent être compatibles entre elles.

Depuis le dossier du projet, lancer dans PowerShell :

```powershell
python prepare_dataset.py --root .\archive\ModelNet10 --out .\dataset
```

Pour choisir deux autres classes, par exemple `bed` et `sofa` :

```powershell
python prepare_dataset.py --root .\archive\ModelNet10 --out .\dataset --classes bed sofa
```

Les images sont enregistrées dans `dataset\raw` (profondeur) et `dataset\lbp` (descripteurs LBP), selon l'arborescence `partition\classe`. Le script ne supprime pas les images des autres classes déjà présentes. Un cache par classe et partition évite les reconversions lorsque les paramètres n'ont pas changé ; il est invalidé si la taille des vues ou le nombre de points change.

## Entraînement et évaluation

Entraîner le modèle sur les mosaïques LBP :

```powershell
python train.py --data .\dataset\lbp --classes chair table
```

Pour utiliser les projections de profondeur sans LBP :

```powershell
python train.py --data .\dataset\raw --classes chair table
```

Le jeu d'entraînement est séparé de façon stratifiée en sous-ensembles d'entraînement et de validation. La validation choisit le meilleur modèle ; le jeu de test ModelNet10 n'est utilisé qu'une seule fois pour l'évaluation finale. Les résultats et la galerie des erreurs sont écrits dans `dataset\lbp_results` ou `dataset\raw_results`.

Pour conserver exactement deux classes, `--classes` doit en indiquer deux, et les deux doivent exister dans les partitions d'entraînement et de test.
