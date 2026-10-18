'''
Outils communs aux notebooks du TP « Généralisation dans les modèles de diffusion »
(Kadkhodaie, Guth, Simoncelli, Mallat, ICLR 2024).

Ce module (dossier tp/ du fork) complète le code des auteurs (dossier code/) sans le modifier :
 - données : images C^alpha (générateur des auteurs), CelebA reconstruit ;
 - débruiteur optimal pour une loi empirique (mémorisation exacte) ;
 - échantillonneur de l'article (algorithme 1, Kadkhodaie & Simoncelli 2021) traité par lots ;
 - jacobien d'un débruiteur par lots de produits vecteur-jacobien ;
 - entraînement de petits UNet sans biais ;
 - mesures (PSNR, similarité cosinus) et affichages.

Conventions des auteurs reprises ici : images dans [0, 1], de forme (B, 1, K, K) ; les réseaux
renvoient le RÉSIDU r(y) = y - f(y) (le bruit estimé), et non l'image débruitée f(y).
'''
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # racine du fork
DONNEES = os.path.join(RACINE, 'tp', 'donnees')                           # données reconstruites (ignoré par git)
if os.path.join(RACINE, 'code') not in sys.path:
    sys.path.insert(0, os.path.join(RACINE, 'code'))                     # code des auteurs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def retablir_affichage():
    '''Plusieurs modules des auteurs imposent matplotlib.use('Agg'), qui empêche l'affichage des figures
    dans un notebook : on rétablit l'affichage en ligne si l'on est dans IPython.'''
    try:
        ipython = get_ipython()  # noqa: F821 (défini par IPython)
    except NameError:
        return
    if ipython is not None:
        ipython.run_line_magic('matplotlib', 'inline')


import dataloader_func  # noqa: E402,F401  (impose Agg)
import synthetic_data_generators  # noqa: E402,F401  (impose Agg)
retablir_affichage()


def fixer_graine(graine):
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)


# ------------------------------------------------------------------------------------------------
# Données
# ------------------------------------------------------------------------------------------------
def images_calpha(alpha, n, taille=80, graine=0):
    '''n images C^alpha (contour C^alpha entre deux fonds C^alpha), comme celles des auteurs.'''
    from synthetic_data_generators import make_C_alpha_images
    fixer_graine(graine)
    return make_C_alpha_images(alpha=alpha, beta=alpha, separable=False, im_size=taille, num_samples=n,
                               constant_background=False, factor=(2, 2), antialiasing=0)


def charger_celeba(taille=80, n=2 * 99325):
    '''CelebA en niveaux de gris, taille x taille, reconstruit au premier appel (voir prepare_celeba80x80.py).'''
    from prepare_celeba80x80 import build_celeba
    if taille == 80 and n == 2 * 99325:
        chemin = os.path.join(DONNEES, 'img_align_celeba', 'train80x80_no_repeats.pt')
    else:
        chemin = os.path.join(DONNEES, 'img_align_celeba', 'celeba{0}x{0}_{1}.pt'.format(taille, n))
    if not os.path.exists(chemin):
        build_celeba(chemin, n_images=n, taille=taille)
    return torch.load(chemin)


def charger_unet_auteurs(nom_donnees, taille_ensemble, echange=False):
    '''Charge un débruiteur fourni par les auteurs (UNet 3 blocs, 80x80), en mode évaluation.'''
    from model_loader_func import load_UNet
    modele = load_UNet(base_path=os.path.join(RACINE, 'denoisers', 'UNet'), training_data_name=nom_donnees,
                       training_noise='0to255', RF=90, set_size=str(taille_ensemble), swap=echange)
    return modele.to(DEVICE).eval()


# ------------------------------------------------------------------------------------------------
# Débruiteur optimal pour une loi empirique
# ------------------------------------------------------------------------------------------------
def distances_carrees(y, X):
    '''|y_b - x_i|^2 pour toutes les paires, en double précision : les poids a posteriori dépendent
    de petites différences entre de grandes distances, que la simple précision arrondit.'''
    y, X = y.flatten(1).double(), X.flatten(1).double()
    return ((y ** 2).sum(1, keepdim=True) + (X ** 2).sum(1)[None, :] - 2 * y @ X.T).clamp_min(0)


def poids_empiriques(y, X, sigma):
    '''Poids a posteriori w_i(y) = P(x = x_i | y) pour la loi uniforme sur les images X (N, 1, K, K).

    w_i(y) est proportionnel à exp(-|y - x_i|^2 / 2 sigma^2) : softmax des distances.
    sigma : scalaire ou tenseur (B,).'''
    d2 = distances_carrees(y, X)
    sigma = torch.as_tensor(sigma, dtype=d2.dtype, device=y.device).reshape(-1, 1)
    return torch.softmax(-d2 / (2 * sigma ** 2), dim=1).to(y.dtype)


def debruiteur_empirique(y, X, sigma):
    '''E[x | y] pour la loi uniforme sur X : moyenne des x_i pondérée par w_i(y).'''
    w = poids_empiriques(y, X, sigma)
    return (w @ X.flatten(1)).view_as(y)


def jacobien_empirique(y, X, sigma):
    '''Jacobien exact du débruiteur empirique en y (1, 1, K, K) : Cov_w[x] / sigma^2, matrice (d, d).'''
    w = poids_empiriques(y, X, sigma)[0]
    Xf = X.flatten(1)
    moyenne = w @ Xf
    centre = (Xf - moyenne) * w.sqrt()[:, None]
    return centre.T @ centre / sigma ** 2


class ResiduEmpiriqueAveugle(nn.Module):
    '''Débruiteur empirique « aveugle », présenté comme les réseaux des auteurs (il renvoie le résidu).

    Le niveau de bruit n'est pas donné : il est estimé par sigma^2 = min_i |y - x_i|^2 / d (distance
    à l'image d'entraînement la plus proche), ce qui permet de l'utiliser dans l'échantillonneur.'''

    def __init__(self, X):
        super().__init__()
        self.X = X

    def forward(self, y):
        d2 = distances_carrees(y, self.X)
        sigma2 = d2.min(dim=1, keepdim=True).values / y[0].numel()
        w = torch.softmax(-d2 / (2 * sigma2.clamp_min(1e-12)), dim=1).to(y.dtype)
        return y - (w @ self.X.flatten(1)).view_as(y)


# ------------------------------------------------------------------------------------------------
# Échantillonnage (algorithme 1 de l'article)
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def echantillonner(residu, init, sigma_fin=0.01, h0=0.01, beta=0.1, max_iter=5000, graine=0,
                   garder_trajectoire=False, pas_trajectoire=10, sigma_max=2.0):
    '''Échantillonnage par montée stochastique du gradient de log p, par lots.

    Reprend univ_inv_sol (code des auteurs, cas de la synthèse) : à chaque pas,
        r = residu(y)                       (r = y - f(y) = -sigma^2 * score)
        sigma = |r| / sqrt(d)               (niveau de bruit effectif, estimé)
        h_t = h0 t / (1 + h0 (t - 1))       (pas croissant)
        gamma = sigma sqrt((1 - beta h)^2 - (1 - h)^2)
        y <- y - h r + gamma z
    jusqu'à sigma < sigma_fin, puis un dernier débruitage y - r(y).
    Le bruit z est tiré pour tout le lot à chaque pas : deux appels avec la même graine et le même
    init utilisent les mêmes tirages, même avec des débruiteurs différents (comparaison A/B).
    init : (B, 1, K, K), par exemple moyenne + N(0, sigma0^2).
    Comme chez les auteurs (« not converging »), un échantillon dont le niveau de bruit estimé dépasse
    sigma_max est abandonné : il est rendu rempli de NaN et signalé dans infos['divergents'].'''
    torch.manual_seed(graine)
    y = init.clone().to(DEVICE)
    B, d = y.shape[0], y[0].numel()
    actif = torch.ones(B, dtype=torch.bool, device=DEVICE)
    diverge = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    sigmas, trajectoire = [], []
    t = 1
    while actif.any() and t <= max_iter:
        h = h0 * t / (1 + h0 * (t - 1))
        if garder_trajectoire and (t - 1) % pas_trajectoire == 0:
            trajectoire.append(y.cpu().clone())        # y avant le pas t, de niveau de bruit sigmas[t - 1]
        z = torch.randn_like(y)
        r = residu(y[actif])
        sigma = r.flatten(1).norm(dim=1) / np.sqrt(d)
        gamma = sigma * np.sqrt((1 - beta * h) ** 2 - (1 - h) ** 2)
        y[actif] = y[actif] - h * r + gamma.view(-1, 1, 1, 1) * z[actif]
        s = torch.full((B,), float('nan'), device=DEVICE)
        s[actif] = sigma
        sigmas.append(s.cpu())
        diverge[actif.clone()] = sigma > sigma_max
        actif[actif.clone()] = (sigma > sigma_fin) & (sigma <= sigma_max)
        t += 1
    final = y - residu(y)
    final[diverge] = float('nan')
    infos = {'sigmas': torch.stack(sigmas), 'iterations': t - 1, 'divergents': diverge.cpu()}
    if garder_trajectoire:
        infos['trajectoire'] = torch.stack(trajectoire)
    return final, infos


def init_echantillons(moyenne, n, sigma0=1.0, graine=0):
    '''Images initiales de l'échantillonneur : moyenne + N(0, sigma0^2), comme chez les auteurs.'''
    g = torch.Generator().manual_seed(graine)
    return moyenne.reshape(1, *moyenne.shape[-3:]).cpu() + sigma0 * torch.randn(n, *moyenne.shape[-3:], generator=g)


# ------------------------------------------------------------------------------------------------
# Jacobien d'un débruiteur
# ------------------------------------------------------------------------------------------------
def jacobien(f, y, taille_lot=256):
    '''Jacobien (d, d) de f en y (1, 1, K, K) : J[i, j] = d f_i / d y_j.

    Calculé par lots de produits vecteur-jacobien (torch.func.vjp + vmap), beaucoup plus vite que
    calc_jacobian des auteurs (un gradient par pixel).'''
    from torch.func import vjp, vmap
    y = y.detach().to(DEVICE)
    sortie, fvjp = vjp(f, y)
    d = y.numel()
    lignes = []
    for debut in range(0, d, taille_lot):
        n = min(taille_lot, d - debut)
        E = torch.zeros(n, d, device=DEVICE)
        E[torch.arange(n), debut + torch.arange(n)] = 1
        (g,) = vmap(fvjp)(E.view(n, *sortie.shape))
        lignes.append(g.reshape(n, d).detach())
    return torch.cat(lignes)


def vecteurs_propres_principaux(f, y, k=40, tol=0.0):
    '''Les k plus grandes valeurs propres (et vecteurs propres) de la partie symétrique (J + J^T)/2
    du jacobien de f en y, sans former J : méthode de Lanczos (scipy eigsh), chaque produit
    matrice-vecteur coûtant un produit jacobien-vecteur (jvp) et un vecteur-jacobien (vjp).
    tol : tolérance de eigsh (0 = précision machine ; 1e-4 suffit pour des figures et va bien plus vite
    quand de nombreuses valeurs propres sont proches les unes des autres).'''
    from scipy.sparse.linalg import LinearOperator, eigsh
    from torch.func import jvp, vjp
    y = y.detach().to(DEVICE)
    _, fvjp = vjp(f, y)
    d = y.numel()

    def produit(v):
        v = torch.as_tensor(np.asarray(v, dtype=np.float32).reshape(y.shape), device=DEVICE)
        jv = jvp(f, (y,), (v,))[1]
        jtv = fvjp(v)[0]
        return ((jv + jtv) / 2).detach().flatten().cpu().numpy().astype(np.float64)

    valeurs, vecteurs = eigsh(LinearOperator((d, d), matvec=produit, dtype=np.float64), k=k, which='LA', tol=tol)
    ordre = np.argsort(valeurs)[::-1]
    return torch.tensor(valeurs[ordre].copy()), torch.tensor(vecteurs[:, ordre].copy(), dtype=torch.float32)


def trace_jacobien(f, y, n_sondes=32, graine=0):
    '''Estimation de tr J(y) par la méthode de Hutchinson : tr J = E[v^T J v] pour v à composantes
    ±1 indépendantes ; chaque sonde coûte un produit jacobien-vecteur (jvp). Renvoie (moyenne, écart-type
    de la moyenne).'''
    from torch.func import jvp
    g = torch.Generator().manual_seed(graine)
    y = y.detach().to(DEVICE)
    V = (torch.randint(0, 2, (n_sondes, *y.shape[1:]), generator=g) * 2 - 1).float().to(DEVICE)
    Jv = jvp(f, (y.repeat(n_sondes, 1, 1, 1),), (V,))[1]
    valeurs = (V * Jv).flatten(1).sum(dim=1).detach().cpu()
    return valeurs.mean().item(), valeurs.std().item() / np.sqrt(n_sondes)


def asymetrie_jacobien(f, y, n_sondes=8, graine=0):
    '''|J v - J^T v| / |J v| moyen sur des directions v aléatoires (nul si J est symétrique).'''
    from torch.func import jvp, vjp
    g = torch.Generator().manual_seed(graine)
    y = y.detach().to(DEVICE)
    _, fvjp = vjp(f, y)
    rapports = []
    for _ in range(n_sondes):
        v = torch.randn(y.shape, generator=g).to(DEVICE)
        jv = jvp(f, (y,), (v,))[1]
        jtv = fvjp(v)[0]
        rapports.append(((jv - jtv).norm() / jv.norm()).item())
    return float(np.mean(rapports))


def debruiteur_depuis_residu(modele):
    '''f(y) = y - r(y) pour un réseau des auteurs (qui renvoie le résidu r).'''
    return lambda y: y - modele(y)


def decomposition_jacobien(J):
    '''Valeurs et vecteurs propres de la partie symétrique de J, par valeurs décroissantes,
    et asymétrie relative |J - J^T| / |J| (nulle pour le débruiteur optimal).'''
    asymetrie = (torch.linalg.norm(J - J.T) / torch.linalg.norm(J)).item()
    valeurs, vecteurs = torch.linalg.eigh((J + J.T) / 2)
    ordre = torch.argsort(valeurs, descending=True)
    return valeurs[ordre], vecteurs[:, ordre], asymetrie


# ------------------------------------------------------------------------------------------------
# Petits UNet et entraînement
# ------------------------------------------------------------------------------------------------
def petit_unet(nb_blocs=2, nb_noyaux=32):
    '''UNet sans biais des auteurs (network.UNet), réduit : nb_blocs blocs encodeur-décodeur,
    nb_noyaux canaux au premier niveau (64 et 3 blocs pour les modèles de l'article).'''
    import argparse
    from model_loader_func import init_UNet
    from dataloader_func import weights_init_kaiming
    rf = {1: 18, 2: 45, 3: 90, 4: 180}[nb_blocs]
    modele = init_UNet(my_args=argparse.Namespace(RF=rf, num_kernels=nb_noyaux))
    modele.apply(weights_init_kaiming)  # initialisation des auteurs (main.py)
    return modele.to(DEVICE)


def nombre_parametres(modele):
    return sum(p.numel() for p in modele.parameters())


def bruiter(x, sigma_max=1.0, quadratique=True, sigma=None):
    '''y = x + sigma z. Sans sigma imposé : sigma tiré par image comme chez les auteurs
    (add_noise_torch, quadratic=True : sigma = sigma_max u^2, u uniforme sur [0, 1]).'''
    if sigma is None:
        u = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        sigma = sigma_max * (u ** 2 if quadratique else u)
    bruit = torch.randn_like(x) * sigma
    return x + bruit, bruit


def entrainer(modele, X, n_iterations=2000, taille_lot=128, lr=1e-3, graine=0, sigma_max=1.0,
              afficher_tous=500, quadratique=True):
    '''Entraîne un réseau à prédire le bruit (perte des auteurs : somme des carrés / taille du lot).

    Les lots sont tirés avec remise dans X : le nombre de pas ne dépend pas de N (de même, les auteurs
    répètent les images jusqu'à 250 000 par époque, sur 1000 époques de lots de 512). Le pas
    d'apprentissage est divisé par 2 aux quarts du parcours (chez les auteurs : toutes les 100 époques).
    quadratique=False tire sigma uniformément sur [0, sigma_max] (plus de grands bruits que chez les auteurs).'''
    fixer_graine(graine)
    X = X.to(DEVICE)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=lr)
    planif = torch.optim.lr_scheduler.MultiStepLR(
        optimiseur, [n_iterations // 4, n_iterations // 2, 3 * n_iterations // 4], gamma=0.5)
    pertes = []
    debut = time.time()
    modele.train()
    for it in range(1, n_iterations + 1):
        x = X[torch.randint(len(X), (taille_lot,), device=DEVICE)]
        y, bruit = bruiter(x, sigma_max, quadratique)
        perte = ((modele(y) - bruit) ** 2).sum() / taille_lot
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        planif.step()
        pertes.append(perte.item())
        if afficher_tous and it % afficher_tous == 0:
            print('  itération {:5d}  perte {:8.3f}  ({:.0f} s)'.format(it, np.mean(pertes[-afficher_tous:]),
                                                                       time.time() - debut))
    modele.eval()
    return pertes


# ------------------------------------------------------------------------------------------------
# Mesures
# ------------------------------------------------------------------------------------------------
def psnr(x, x_chapeau, amplitude=1.0):
    '''PSNR de chaque image (dB) : 10 log10(amplitude^2 / EQM).'''
    eqm = ((x - x_chapeau) ** 2).flatten(1).mean(dim=1)
    return 10 * torch.log10(amplitude ** 2 / eqm)


def psnr_entree(sigma, amplitude=1.0):
    '''PSNR de l'image bruitée : 10 log10(amplitude^2 / sigma^2).'''
    return 10 * np.log10(amplitude ** 2 / np.asarray(sigma, dtype=float) ** 2)


@torch.no_grad()
def courbe_psnr(debruiteur, X, sigmas, graine=0, taille_lot=200):
    '''PSNR de sortie moyen (dB) de f = debruiteur sur les images X, pour chaque sigma.
    debruiteur(y, sigma) renvoie l'image débruitée.'''
    fixer_graine(graine)
    resultats = []
    for sigma in sigmas:
        valeurs = []
        for debut in range(0, len(X), taille_lot):
            x = X[debut:debut + taille_lot].to(DEVICE)
            y = x + sigma * torch.randn_like(x)
            valeurs.append(psnr(x, debruiteur(y, sigma)).cpu())
        resultats.append(torch.cat(valeurs).mean().item())
    return np.array(resultats)


def similarites_cosinus(A, B, centrer=True):
    '''Matrice des similarités cosinus entre les images de A (n, ...) et de B (m, ...),
    après retrait de la moyenne de chaque image si centrer (comme im_set_corr des auteurs).'''
    A, B = A.flatten(1).float(), B.flatten(1).float()
    if centrer:
        A = A - A.mean(dim=1, keepdim=True)
        B = B - B.mean(dim=1, keepdim=True)
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return A @ B.T


@torch.no_grad()
def plus_proches_voisins(A, B, centrer=True, taille_lot=8192):
    '''Pour chaque image de A, similarité cosinus maximale avec les images de B et son indice.
    B peut être grand (CelebA entier) : il est parcouru par morceaux.'''
    A = A.to(DEVICE)
    meilleures = torch.full((len(A),), -2.0, device=DEVICE)
    indices = torch.zeros(len(A), dtype=torch.long, device=DEVICE)
    for debut in range(0, len(B), taille_lot):
        s = similarites_cosinus(A, B[debut:debut + taille_lot].to(DEVICE), centrer)
        valeurs, idx = s.max(dim=1)
        mieux = valeurs > meilleures
        meilleures[mieux] = valeurs[mieux]
        indices[mieux] = idx[mieux] + debut
    return meilleures.cpu(), indices.cpu()


def groupes_distincts(images, seuil=0.95):
    '''Regroupe les images presque identiques (similarité cosinus > seuil) : renvoie le numéro
    de groupe de chaque image (regroupement glouton). Peu de groupes = peu de diversité.'''
    S = similarites_cosinus(images, images)
    groupe = -torch.ones(len(images), dtype=torch.long)
    k = 0
    for i in range(len(images)):
        if groupe[i] < 0:
            groupe[(S[i] > seuil) & (groupe < 0)] = k
            k += 1
    return groupe


# ------------------------------------------------------------------------------------------------
# Affichage (toute carte colorée porte sa barre graduée)
# ------------------------------------------------------------------------------------------------
NOTEBOOK = None   # nom du notebook courant : fixé par nommer_figures


def nommer_figures(nom_notebook, dossier='resultats/figures'):
    '''Active l'enregistrement des figures nommées : montrer(nom) écrit dossier/<notebook>_fig_<nom>.png.
    Ces fichiers sont ceux qu'inclut l'énoncé du TP.'''
    global NOTEBOOK, DOSSIER_FIGURES
    NOTEBOOK, DOSSIER_FIGURES = nom_notebook, dossier
    os.makedirs(dossier, exist_ok=True)


def montrer(nom=None, fig=None):
    '''Affiche la figure courante ; si nom est donné et nommer_figures appelé, l'enregistre aussi.'''
    fig = fig or plt.gcf()
    if nom is not None and NOTEBOOK is not None:
        fig.savefig(os.path.join(DOSSIER_FIGURES, '{}_fig_{}.png'.format(NOTEBOOK, nom)), dpi=130,
                    bbox_inches='tight')
    plt.show()

def afficher_images(images, titres=None, ncol=10, vmin=0.0, vmax=1.0, titre=None, taille=1.5,
                    etiquette='intensité', nom=None):
    '''Grille d'images en niveaux de gris, bornes communes [vmin, vmax] : une seule barre graduée.'''
    images = images.detach().cpu().reshape(-1, *images.shape[-2:])
    n = len(images)
    nlig = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nlig, ncol, figsize=(taille * ncol + 0.8, taille * nlig + (0.4 if titre else 0)),
                             squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis('off')
        if i < n:
            im = ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
            if titres is not None:
                ax.set_title(titres[i], fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=etiquette + ' (bornes communes)')
    if titre:
        fig.suptitle(titre)
    montrer(nom, fig)


def afficher_vecteurs(vecteurs, K, titres=None, ncol=7, taille=1.9, titre=None, nom=None):
    '''Vecteurs de R^(K*K) affichés en cartes K x K (rouge/bleu = signe). Chaque carte est
    normalisée pour elle-même (centrée en 0) : une barre graduée par carte.'''
    vecteurs = vecteurs.detach().cpu().reshape(-1, K, K)
    n = len(vecteurs)
    nlig = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nlig, ncol, figsize=(taille * ncol, taille * nlig + (0.4 if titre else 0)),
                             squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis('off')
        if i < n:
            m = vecteurs[i].abs().max().item()
            im = ax.imshow(vecteurs[i], cmap='RdBu_r', vmin=-m, vmax=m)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            if titres is not None:
                ax.set_title(titres[i], fontsize=8)
    if titre:
        fig.suptitle(titre + ' (une échelle par carte)')
    fig.tight_layout()
    montrer(nom, fig)


def trace_psnr(courbes, sigmas, titre='', identite=True, ax=None):
    '''Courbes PSNR de sortie en fonction du PSNR d'entrée. courbes : {étiquette: valeurs}.'''
    x = psnr_entree(sigmas)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    for etiquette, valeurs in courbes.items():
        ax.plot(x, valeurs, '.-', label=etiquette)
    if identite:
        lim = [min(x.min(), 0), x.max()]
        ax.plot(lim, lim, 'k--', lw=0.8, label='identité')
    ax.set_xlabel("PSNR d'entrée (dB)")
    ax.set_ylabel('PSNR de sortie (dB)')
    ax.set_title(titre)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def pente(x, y, x_min=None):
    '''Pente de la droite des moindres carrés (sur les points d'abscisse >= x_min).'''
    x, y = np.asarray(x), np.asarray(y)
    garde = np.ones_like(x, dtype=bool) if x_min is None else x >= x_min
    return np.polyfit(x[garde], y[garde], 1)[0]
