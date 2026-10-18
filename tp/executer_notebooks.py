'''
Exécute des notebooks du TP et affiche leur durée ; les figures nommées sont écrites par les notebooks
eux-mêmes dans tp/resultats/figures/ (tp_outils.nommer_figures).

Usage :
    python executer_notebooks.py [--refaire] TP_debruiteur_empirique.ipynb [autres notebooks...]

Chaque notebook exécuté, avec ses sorties, est enregistré dans tp/resultats/notebooks_executes/. Reprise après une
interruption : un notebook dont la version exécutée est plus récente que la source et ne contient pas d'erreur est
sauté, sauf avec --refaire. Les erreurs de cellule sont affichées sans interrompre l'exécution ; un journal est tenu dans
tp/resultats/journal_execution.txt.
'''
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ICI = os.path.dirname(os.path.abspath(__file__))
SORTIE = os.path.join(ICI, 'resultats', 'notebooks_executes')
# le noyau « python3 » lance « python » : celui de l'interpréteur qui exécute ce script
os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH']
ERREURS_ADMISES = set()   # noms d'erreurs tolérées dans une version exécutée « à jour »


def erreurs(nb):
    return [(i, o.ename, o.evalue) for i, c in enumerate(nb.cells) if c.cell_type == 'code'
            for o in c.get('outputs', []) if o.output_type == 'error']


def a_jour(chemin):
    execute = os.path.join(SORTIE, os.path.basename(chemin))
    if not os.path.exists(execute) or os.path.getmtime(execute) < os.path.getmtime(chemin):
        return False
    return all(nom in ERREURS_ADMISES for _, nom, _ in erreurs(nbformat.read(execute, as_version=4)))


def journal(ligne):
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(os.path.join(ICI, 'resultats', 'journal_execution.txt'), 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + ligne + '\n')


def executer(chemin, refaire=False):
    nom = os.path.basename(chemin)
    if not refaire and a_jour(chemin):
        print('%-45s déjà exécuté (version à jour, sans erreur) : sauté' % nom, flush=True)
        return
    dossier = os.path.dirname(os.path.abspath(chemin))
    nb = nbformat.read(chemin, as_version=4)
    journal('début ' + nom)
    debut = time.time()
    NotebookClient(nb, timeout=12 * 3600, kernel_name='python3', allow_errors=True,
                   resources={'metadata': {'path': dossier}}).execute()
    duree = time.time() - debut
    os.makedirs(SORTIE, exist_ok=True)
    nbformat.write(nb, os.path.join(SORTIE, nom))
    err = erreurs(nb)
    ligne = '%-45s %6.0f s  (%d min %02d s)  erreurs : %d' % (nom, duree, duree // 60, duree % 60, len(err))
    print(ligne, flush=True)
    journal('fin   ' + ligne)
    for i, n, message in err:
        print('    cellule %d : %s : %s' % (i, n, message[:300]))


if __name__ == '__main__':
    arguments = sys.argv[1:]
    refaire = '--refaire' in arguments
    for chemin in [a for a in arguments if a != '--refaire']:
        executer(chemin, refaire)
