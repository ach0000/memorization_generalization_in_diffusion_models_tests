# TP « Mémoriser ou généraliser ? » (Master TRIED)

Travaux pratiques autour de l'article de Kadkhodaie, Guth, Simoncelli et Mallat,
*Generalization in diffusion models arises from geometry-adaptive harmonic representations* (ICLR 2024).
Ce dépôt est un fork du [code des auteurs](https://github.com/LabForComputationalVision/memorization_generalization_in_diffusion_models),
complété pour être utilisé sur Colab.

Chaque notebook commence par une cellule d'installation : sur Colab, elle télécharge le code et les seuls poids
nécessaires ; en local, elle rend le code importable.

| Notebook | Matériel | Colab |
|---|---|---|
| `TP_debruiteur_empirique` : le débruiteur optimal d'un jeu fini mémorise | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/memorization_generalization_in_diffusion_models_tests/blob/main/tp/TP_debruiteur_empirique.ipynb) |
| `TP_gaussien_wiener` : filtre de Wiener et base de Fourier | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/memorization_generalization_in_diffusion_models_tests/blob/main/tp/TP_gaussien_wiener.ipynb) |
| `TP_entrainement_debruiteurs` : entraîner de petits UNet sur des jeux disjoints | GPU (CPU réduit) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/memorization_generalization_in_diffusion_models_tests/blob/main/tp/TP_entrainement_debruiteurs.ipynb) |
| `TP_extraction_memorisation` : retrouver les images mémorisées | GPU (CPU lent) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/memorization_generalization_in_diffusion_models_tests/blob/main/tp/TP_extraction_memorisation.ipynb) |
| `TP_jacobien_base_adaptative` : UNet sans biais, jacobien, base adaptative | CPU ou GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/memorization_generalization_in_diffusion_models_tests/blob/main/tp/TP_jacobien_base_adaptative.ipynb) |
| `auteurs/Demo_UNet_CelebA80x80_colab` (auteurs, adapté) | GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/memorization_generalization_in_diffusion_models_tests/blob/main/tp/auteurs/Demo_UNet_CelebA80x80_colab.ipynb) |
| `auteurs/Demo_UNet_Calpha80x80_colab` (auteurs, adapté) | GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/memorization_generalization_in_diffusion_models_tests/blob/main/tp/auteurs/Demo_UNet_Calpha80x80_colab.ipynb) |

## Ajouts au dépôt des auteurs

Tous les ajouts sont dans ce dossier `tp/` ; **aucun fichier des auteurs n'est modifié** (code, poids, notebooks,
`.gitignore`).

- `TP_*.ipynb` : les notebooks du TP (sans sorties).
- `tp_outils.py` : outils communs aux notebooks `TP_*` (débruiteur empirique, échantillonneur par lots,
  jacobien, entraînement, mesures, figures) ; il rend importable le code des auteurs (`code/`).
- `prepare_celeba80x80.py` : reconstruit CelebA en niveaux de gris (non distribué par les auteurs) dans
  `tp/donnees/`, à partir d'un miroir HuggingFace. L'ordre des images et le retrait des doublons diffèrent de
  ceux des auteurs : on ignore quelles images ont servi à entraîner les débruiteurs fournis (voir
  `TP_extraction_memorisation`).
- `auteurs/<nom>_colab.ipynb` : versions Colab des notebooks des auteurs (`notebooks/<nom>.ipynb`, inchangés) :
  bouton Colab et cellule d'installation ajoutés, `.cuda()` remplacé par `.to(device)`, `%load_ext autoreload`
  retiré (incompatible avec le Python de Colab), CelebA lu dans `tp/donnees/` (reconstruit au premier
  lancement). Les sorties affichées sont celles des auteurs.
- `executer_notebooks.py` : exécute des notebooks et affiche leur durée (`python executer_notebooks.py
  TP_gaussien_wiener.ipynb`) ; les notebooks exécutés et les figures nommées sont écrits dans `tp/resultats/`.
- `enonce/generalisation_diffusion.pdf` : l'énoncé.
- `resultats/` et `donnees/` (résultats locaux et données reconstruites) sont ignorés par git.
