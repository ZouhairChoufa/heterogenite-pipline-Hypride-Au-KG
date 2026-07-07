"""
_02_matching_sbert_only.py
==========================
VARIANTE A — SBERT Only (Ablation Study)

Utilise UNIQUEMENT Sentence-BERT pour le schema matching.
Aucun appel LLM. Les colonnes sous le seuil τ sont forcées
vers le meilleur candidat SBERT, même si la confiance est faible.

Objectif : quantifier la dégradation de précision quand on supprime
le LLM de l'architecture hybride.

Usage :
    python _02_matching_sbert_only.py
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from utils.logger import get_logger

logger = get_logger("ablation.sbert_only")


# ==============================================================================
# Ontologie cible (identique au module hybride pour comparaison équitable)
# ==============================================================================

ONTOLOGIE_CIBLE: dict[str, str] = {
    "ex:employeeId":          "Identifiant unique de l'employé ou du client",
    "schema:birthDate":       "Date de naissance de la personne",
    "foaf:familyName":        "Nom de famille de la personne",
    "foaf:firstName":         "Prénom de la personne",
    "ex:fullName":            "Nom complet (prénom + nom de famille concaténés)",
    "schema:email":           "Adresse email de contact",
    "ex:department":          "Département ou service dans l'organisation",
    "ex:annualRevenue":       "Chiffre d'affaires ou revenu annuel",
    "schema:jobTitle":        "Poste ou type de contrat professionnel",
    "schema:addressLocality": "Ville ou localité",
}

# ==============================================================================
# Gold Standard — vérité terrain pour l'évaluation
# Clé : nom de colonne source → valeur : URI ontologique attendue
# ==============================================================================

GOLD_STANDARD: dict[str, Optional[str]] = {
    # Colonnes Source A — CRM
    "ID_Client":    "ex:employeeId",
    "nom_complet":  "ex:fullName",
    "email":        "schema:email",
    "dt_naissance": "schema:birthDate",
    "ville":        "schema:addressLocality",
    "ca_annuel":    "ex:annualRevenue",
    "segment":      None,               # Pas de correspondance → UNMATCHED attendu
    # Colonnes Source B — RH
    "Matricule_RH":    "ex:employeeId",
    "Nom_Famille":     "foaf:familyName",
    "Prenom":          "foaf:firstName",
    "date_nais":       "schema:birthDate",
    "departement":     "ex:department",
    "contrat":         "schema:jobTitle",
    "salaire_mensuel": None,            # Pas de correspondance → UNMATCHED attendu
    "date_embauche":   None,            # Pas de correspondance → UNMATCHED attendu
}


# ==============================================================================
# Évaluateur
# ==============================================================================

def compute_metrics(
    results: list[dict],
    gold: dict[str, Optional[str]]
) -> dict:
    """
    Calcule Précision, Rappel, F1 et le taux de colonnes correctement rejetées.

    Logique :
      - Vrai Positif (TP)  : colonne mappée → bonne URI et gold != None
      - Faux Positif (FP)  : colonne mappée → mauvaise URI  OU gold == None (rejet attendu)
      - Faux Négatif (FN)  : colonne UNMATCHED alors que gold != None
      - Vrai Négatif (TN)  : colonne UNMATCHED et gold == None  ✓
    """
    tp = fp = fn = tn = 0

    for r in results:
        col        = r["source_column"]
        predicted  = r["target_uri"]
        expected   = gold.get(col)          # None si colonne sans correspondance

        if expected is None:
            # Cas où on attend un UNMATCHED
            if predicted == "UNMATCHED":
                tn += 1                     # Rejet correct
            else:
                fp += 1                     # Fausse correspondance
        else:
            # Cas où on attend une correspondance réelle
            if predicted == "UNMATCHED":
                fn += 1                     # Manqué
            elif predicted == expected:
                tp += 1                     # Bonne correspondance
            else:
                fp += 1                     # Mauvaise correspondance

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    tnr       = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Spécificité (taux rejet correct)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "specificity": round(tnr, 4),
    }


# ==============================================================================
# Classe principale — SBERT Only
# ==============================================================================

class SBERTOnlyMatcher:
    """
    Schema Matcher basé exclusivement sur Sentence-BERT.

    Comportement :
      - Encode toutes les colonnes sources en embeddings
      - Calcule la similarité cosinus avec les propriétés ontologiques
      - Accepte systématiquement le meilleur candidat, SANS recours au LLM
      - Les colonnes sous le seuil τ sont quand même forcées (best-effort)
        afin d'évaluer le comportement sans filet de sécurité LLM
    """

    def __init__(
        self,
        ontology: dict[str, str],
        model_name: str   = "paraphrase-multilingual-MiniLM-L12-v2",
        tau: float        = 0.82,
        output_path: Path = Path("outputs/ablation_sbert_only.json"),
    ) -> None:
        self.ontology    = ontology
        self.tau         = tau
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.targets = list(ontology.keys())
        self.descs   = list(ontology.values())

        logger.info(f"Chargement SBERT : '{model_name}'...")
        self._model           = SentenceTransformer(model_name)
        self._onto_embeddings = self._model.encode(self.descs)
        logger.info(f"SBERT prêt. Ontologie encodée ({len(self.targets)} propriétés).")

    def match(self, df: pd.DataFrame) -> list[dict]:
        """
        Lance le matching SBERT-only sur toutes les colonnes.

        Retourne une liste de dicts compatibles avec le format du module hybride.
        """
        columns     = df.columns.tolist()
        col_embeds  = self._model.encode(columns)
        cos_scores  = util.cos_sim(col_embeds, self._onto_embeddings)
        results     = []

        for i, col in enumerate(columns):
            best_idx   = int(cos_scores[i].argmax())
            best_score = float(cos_scores[i][best_idx])
            target     = self.targets[best_idx]

            above_tau = best_score >= self.tau

            results.append({
                "source_column": col,
                "target_uri":    target,      # FORCÉ même sous τ (pas de LLM de secours)
                "confidence":    round(best_score, 4),
                "method":        "SBERT",
                "above_tau":     above_tau,
                "justification": (
                    f"SBERT cosinus={best_score:.4f} "
                    f"({'≥' if above_tau else '<'} τ={self.tau})"
                ),
            })

            icon = "✅" if above_tau else "⚠️ "
            logger.info(
                f"  {icon} '{col}' → {target} "
                f"(score={best_score:.3f}, {'au-dessus' if above_tau else 'EN-DESSOUS'} du seuil)"
            )

        return results

    def run(self, source_path: str = "data/Source_B_RH.xlsx") -> dict:
        """
        Exécution complète : matching + évaluation + persistance.

        Returns:
            Dictionnaire complet des résultats et métriques.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("ABLATION A — SBERT Only")
        logger.info("=" * 60)

        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : '{source_path}'")

        df = (
            pd.read_excel(path, engine="openpyxl")
            if path.suffix == ".xlsx"
            else pd.read_csv(path, sep=";")
        )
        logger.info(f"Source chargée : {df.shape} — Colonnes : {df.columns.tolist()}")

        results  = self.match(df)
        metrics  = compute_metrics(results, GOLD_STANDARD)
        elapsed  = round(time.time() - start, 2)

        output = {
            "variant":       "SBERT_Only",
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s":     elapsed,
            "total_columns": len(results),
            "llm_calls":     0,
            "results":       results,
            "metrics":       metrics,
        }

        self.output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("-" * 60)
        logger.info(f"Précision : {metrics['precision']:.4f}")
        logger.info(f"Rappel    : {metrics['recall']:.4f}")
        logger.info(f"F1-score  : {metrics['f1']:.4f}")
        logger.info(f"Temps     : {elapsed}s  |  Appels LLM : 0")
        logger.info(f"Résultats → '{self.output_path}'")
        logger.info("=" * 60)

        return output


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    matcher = SBERTOnlyMatcher(
        ontology=ONTOLOGIE_CIBLE,
        tau=0.82,
        output_path=Path("outputs/ablation_sbert_only.json"),
    )
    matcher.run("data/Source_B_RH.xlsx")
