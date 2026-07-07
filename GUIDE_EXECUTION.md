# Guide d'exécution — PFE Knowledge Graph sur Lightning.ai

## Prérequis

- Compte Lightning.ai (gratuit) : https://lightning.ai
- Clé API Groq (gratuite, 30 req/min) : https://console.groq.com

---

## Étape 1 — Créer un Studio Lightning.ai

1. Connectez-vous sur **lightning.ai**
2. Cliquez sur **"New Studio"**
3. Choisissez le template **"Blank"** (pas de Jupyter, on travaille en terminal)
4. Sélectionnez la machine **CPU · 4 vCPU · 15 GB RAM** (gratuit, suffisant)
5. Cliquez **"Start"** et attendez que le studio s'ouvre

---

## Étape 2 — Uploader le projet

### Option A — Via l'interface (recommandé pour la soutenance)

1. Dans le panneau gauche de Lightning.ai, cliquez sur l'icône **Fichiers**
2. Cliquez **"Upload"** et sélectionnez le dossier `pfe_knowledge_graph/`
3. Vérifiez que tous les fichiers sont présents dans `/teamspace/studios/this_studio/`

### Option B — Via Git (recommandé en pratique)

```bash
# Dans le terminal Lightning.ai
git clone https://github.com/VOTRE_USERNAME/pfe_knowledge_graph.git
cd pfe_knowledge_graph
```

---

## Étape 3 — Ouvrir le terminal

1. Dans Lightning.ai, cliquez sur **"Terminal"** (icône en bas de l'écran)
   ou faites `Ctrl + ~`
2. Vous obtenez un terminal bash Ubuntu standard

```bash
# Vérifiez que vous êtes dans le bon dossier
pwd
# Attendu : /teamspace/studios/this_studio/pfe_knowledge_graph
```

---

## Étape 4 — Créer l'environnement virtuel

```bash
# Créer un venv Python isolé (évite les conflits de dépendances)
python3 -m venv .venv

# Activer le venv
source .venv/bin/activate

# Vérifier (le prompt doit afficher (.venv))
which python
# Attendu : .../pfe_knowledge_graph/.venv/bin/python
```

> **Important** : Réactivez le venv à chaque nouvelle session Lightning.ai avec :
> `source .venv/bin/activate`

---

## Étape 5 — Installer les dépendances

```bash
# Mise à jour de pip
pip install --upgrade pip

# Installation de toutes les dépendances
# (prend 3-5 minutes la première fois)
pip install -r requirements.txt
```

### Vérification de l'installation

```bash
python -c "
import pandas, faker, ydata_profiling, sentence_transformers
import morph_kgc, splink, duckdb, langchain_groq
print('✅ Toutes les dépendances installées avec succès')
"
```

---

## Étape 6 — Configurer la clé API Groq

```bash
# Copier le template d'environnement
cp .env.example .env

# Éditer le fichier .env avec nano
nano .env
```

Dans l'éditeur nano :
1. Remplacez `gsk_VOTRE_CLE_API_GROQ_ICI` par votre vraie clé Groq
2. Sauvegardez : `Ctrl + O` puis `Entrée`
3. Quittez : `Ctrl + X`

### Vérification

```bash
# Teste que la clé est bien lue
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get('GROQ_API_KEY', '')
print('✅ Clé Groq trouvée' if key.startswith('gsk_') else '❌ Clé absente ou invalide')
"
```

---

## Étape 7 — Structure des dossiers

Les dossiers `data/`, `outputs/`, `logs/`, `reports/` sont créés automatiquement
par les scripts. Vérifiez la structure avant de lancer :

```bash
ls -la
# Attendu :
# .env
# .env.example
# .gitignore
# 00_generer_donnees.py
# 01_module_ingestion.py
# 02_module_matching.py
# 03_module_triplification.py
# 04_module_splink.py
# regles_mapping.yml
# requirements.txt
# run_pipeline.py
# utils/
```

---

## Étape 8 — Exécution des modules (dans l'ordre)

### Option A — Pipeline complet automatique (recommandé)

```bash
python run_pipeline.py
```

### Option B — Module par module (pour la démonstration en soutenance)

```bash
# Module 0 : Génère les données factices
python 00_generer_donnees.py
# → data/Source_A_CRM.csv (4 000 lignes)
# → data/Source_B_RH.xlsx (3 000 lignes)

# Module 1 : Profiling
python 01_module_ingestion.py
# → reports/profil_source_A.html
# → reports/profil_source_B.html

# Module 2 : Schema Matching (nécessite la clé Groq)
python 02_module_matching.py
# → outputs/matching_results.json

# Module 3 : Triplification
python 03_module_triplification.py
# → data/Source_B_RH.csv   (conversion automatique)
# → outputs/graphe_connaissances.nt

# Module 4 : Résolution d'entités
python 04_module_splink.py
# → outputs/liens_identite.nt
```

### Option C — Reprendre depuis un module spécifique

```bash
# Si le Module 2 a planté et que 0 et 1 ont déjà tourné :
python run_pipeline.py --from 2

# Tester uniquement le Module 4 :
python run_pipeline.py --only 4
```

---

## Étape 9 — Vérification des résultats

```bash
# Vérifier les fichiers de sortie
ls -lh outputs/
# Attendu :
#   matching_results.json   (résultats du schema matching)
#   graphe_connaissances.nt (triplets RDF)
#   liens_identite.nt       (liens owl:sameAs)

# Compter les triplets RDF
wc -l outputs/graphe_connaissances.nt
# Attendu : ~20 000–25 000 triplets

# Compter les liens owl:sameAs
wc -l outputs/liens_identite.nt
# Attendu : ~500–1 500 liens (selon le chevauchement)

# Consulter les logs détaillés
cat logs/pipeline.log | grep "ERROR\|WARNING\|CRITICAL"

# Voir le résultat du schema matching
python -c "
import json
with open('outputs/matching_results.json') as f:
    r = json.load(f)
print(f'Colonnes matchées : {r[\"total_columns\"]}')
print(f'Via SBERT : {r[\"sbert_count\"]} | Via LLM : {r[\"llm_count\"]} | UNMATCHED : {r[\"unmatched\"]}')
for res in r['results']:
    print(f'  [{res[\"method\"]:10s}] {res[\"source_column\"]:<20} → {res[\"target_uri\"]}')
"
```

---

## Étape 10 — Télécharger les rapports HTML

Pour visualiser les rapports ydata-profiling dans un navigateur :

1. Dans le panneau gauche Lightning.ai → **Fichiers**
2. Naviguez dans `reports/`
3. Clic droit sur `profil_source_A.html` → **"Download"**
4. Ouvrez le fichier téléchargé dans votre navigateur local

---

## Résolution des problèmes courants

| Erreur | Cause probable | Solution |
|--------|---------------|----------|
| `ModuleNotFoundError` | venv non activé | `source .venv/bin/activate` |
| `FileNotFoundError: Source_A_CRM.csv` | Module 0 non exécuté | `python 00_generer_donnees.py` |
| `EnvironmentError: GROQ_API_KEY` | Clé API manquante | Vérifier le fichier `.env` |
| `RateLimitError` (Groq) | Trop de requêtes | Attendre 60s, le retry automatique gère |
| `KeyError: Matricule_RH` | CSV RH absent | Lancer d'abord `python 03_module_triplification.py` |
| `0 liens owl:sameAs` | Seuil trop élevé | Baisser `match_probability_threshold` à 0.75 dans `04_module_splink.py` |

---

## Structure finale du projet

```
pfe_knowledge_graph/
├── .env                          # Clé API (non versionné)
├── .env.example                  # Template de configuration
├── .gitignore
├── requirements.txt
├── regles_mapping.yml            # Règles YARRRML pour Morph-KGC
├── run_pipeline.py               # Orchestrateur principal
│
├── 00_generer_donnees.py         # Module 0 : Données synthétiques
├── 01_module_ingestion.py        # Module 1 : Ingestion & profilage
├── 02_module_matching.py         # Module 2 : Schema matching hybride
├── 03_module_triplification.py   # Module 3 : Triplification RDF
├── 04_module_splink.py           # Module 4 : Résolution d'entités
│
├── utils/
│   ├── __init__.py
│   └── logger.py                 # Logger centralisé (partagé par tous les modules)
│
├── data/                         # Généré automatiquement
│   ├── Source_A_CRM.csv
│   ├── Source_B_RH.xlsx
│   └── Source_B_RH.csv           # Généré par le Module 3
│
├── outputs/                      # Généré automatiquement
│   ├── matching_results.json
│   ├── graphe_connaissances.nt
│   └── liens_identite.nt
│
├── reports/                      # Généré automatiquement
│   ├── profil_source_A.html
│   └── profil_source_B.html
│
└── logs/                         # Généré automatiquement
    └── pipeline.log
```
