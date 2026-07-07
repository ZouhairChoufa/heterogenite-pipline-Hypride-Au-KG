"""
_02_matching_llm_only.py
========================
VARIANTE B — LLM Only (Ablation Study)

Utilise UNIQUEMENT LLaMA 3.1 (via Groq API) pour le schema matching.
Aucun filtrage SBERT préalable — chaque colonne est soumise directement
à l'agent LLM, sans distinction de complexité sémantique.

Objectif : mesurer le coût réel (latence, appels API) d'une approche
purement LLM et comparer sa précision avec la variante hybride.

Usage :
    python _02_matching_llm_only.py
    
Prérequis :
    Variable d'environnement GROQ_API_KEY définie dans .env
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.logger import get_logger

load_dotenv()
logger = get_logger("ablation.llm_only")

try:
    from groq import APIConnectionError, RateLimitError
except ImportError:
    RateLimitError    = Exception
    APIConnectionError = Exception


# ==============================================================================
# Ontologie cible (identique aux deux autres variantes)
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
# Gold Standard (identique à la variante SBERT-Only)
# ==============================================================================

GOLD_STANDARD: dict[str, Optional[str]] = {
    "ID_Client":       "ex:employeeId",
    "nom_complet":     "ex:fullName",
    "email":           "schema:email",
    "dt_naissance":    "schema:birthDate",
    "ville":           "schema:addressLocality",
    "ca_annuel":       "ex:annualRevenue",
    "segment":         None,
    "Matricule_RH":    "ex:employeeId",
    "Nom_Famille":     "foaf:familyName",
    "Prenom":          "foaf:firstName",
    "date_nais":       "schema:birthDate",
    "departement":     "ex:department",
    "contrat":         "schema:jobTitle",
    "salaire_mensuel": None,
    "date_embauche":   None,
}


# ==============================================================================
# Évaluateur (identique variante SBERT-Only pour cohérence)
# ==============================================================================

def compute_metrics(results: list[dict], gold: dict[str, Optional[str]]) -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        col       = r["source_column"]
        predicted = r["target_uri"]
        expected  = gold.get(col)

        if expected is None:
            if predicted == "UNMATCHED":
                tn += 1
            else:
                fp += 1
        else:
            if predicted == "UNMATCHED":
                fn += 1
            elif predicted == expected:
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp)  if (tp + fp) > 0  else 0.0
    recall    = tp / (tp + fn)  if (tp + fn) > 0  else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "specificity": round(tnr, 4),
    }


# ==============================================================================
# Classe principale — LLM Only
# ==============================================================================

class LLMOnlyMatcher:
    """
    Schema Matcher basé exclusivement sur LLaMA 3.1 (Groq API).

    Différence fondamentale avec le module hybride :
      - CHAQUE colonne est soumise au LLM, sans filtrage SBERT préalable
      - L'agent est chargé de décider lui-même si la colonne correspond
        à une propriété ontologique ou doit être rejetée (UNMATCHED)
      - Comptabilisation précise du nombre d'appels API et de la latence
        par colonne (pour démontrer le surcoût vs approche hybride)
    """

    def __init__(
        self,
        ontology: dict[str, str],
        llm_model: str    = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_retries: int   = 4,
        n_samples: int     = 3,
        output_path: Path  = Path("outputs/ablation_llm_only.json"),
    ) -> None:
        self.ontology    = ontology
        self.n_samples   = n_samples
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.targets    = list(ontology.keys())
        self.onto_texte = "\n".join(
            f"  - {uri} : {desc}" for uri, desc in ontology.items()
        )

        self._llm_calls   = 0
        self._call_times  = []     # Latence par appel (ms)
        self._llm_chain   = None
        self._max_retries = max_retries

        # Initialisation immédiate du LLM (pas de lazy loading — on l'utilise toujours)
        self._init_llm(llm_model, temperature)

    def _init_llm(self, model: str, temperature: float) -> None:
        """Initialise le client LLM Groq et la chaîne LangChain."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Variable d'environnement 'GROQ_API_KEY' manquante. "
                "Créez un fichier .env avec GROQ_API_KEY=votre_clé."
            )

        llm    = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        parser = JsonOutputParser()

        prompt = PromptTemplate(
            template=(
                "Tu es un expert en ontologies et web sémantique.\n"
                "Associe la colonne source '{colonne}' "
                "(exemples de valeurs réelles : {exemples}) "
                "à la propriété ontologique la plus appropriée parmi :\n"
                "{ontologie}\n\n"
                "Si AUCUNE propriété ne correspond, retourne 'UNMATCHED' pour 'cible'.\n"
                "Renvoie UNIQUEMENT un objet JSON valide (sans markdown) "
                "avec exactement ces clés :\n"
                "  'cible'        : URI exact de la propriété choisie (ou 'UNMATCHED')\n"
                "  'confiance'    : float entre 0.0 et 1.0\n"
                "  'justification': explication courte (max 20 mots)\n"
                "{format_instructions}"
            ),
            input_variables=["colonne", "exemples", "ontologie"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )
        self._llm_chain = prompt | llm | parser
        logger.info(f"LLM Groq initialisé : '{model}' (temperature={temperature}).")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, 30),
        reraise=True,
    )
    def _call_llm(self, payload: dict) -> dict:
        """Appel LLM avec retry exponentiel et chronométrage."""
        t0     = time.time()
        result = self._llm_chain.invoke(payload)
        latency_ms = round((time.time() - t0) * 1000, 1)
        self._call_times.append(latency_ms)
        self._llm_calls += 1
        logger.debug(f"    Appel LLM #{self._llm_calls} terminé en {latency_ms}ms")
        return result

    def _match_column(self, col: str, df: pd.DataFrame) -> dict:
        """
        Soumet UNE colonne directement au LLM, sans filtrage SBERT.

        Inclut une gestion robuste des cas edge :
          - Colonne quasi-vide
          - URI retournée hors ontologie
          - JSON malformé
        """
        non_null = df[col].dropna()
        n        = min(self.n_samples, len(non_null))

        if n == 0:
            logger.warning(f"  Colonne '{col}' vide — UNMATCHED forcé.")
            return {
                "source_column": col,
                "target_uri":    "UNMATCHED",
                "confidence":    0.0,
                "method":        "LLM_ONLY",
                "justification": "Colonne vide, LLM non invoqué.",
            }

        exemples   = non_null.sample(n, random_state=42).tolist()
        known_uris = set(self.targets) | {"UNMATCHED"}

        payload = {
            "colonne":  col,
            "exemples": json.dumps(exemples, ensure_ascii=False),
            "ontologie": self.onto_texte,
        }

        res = None
        try:
            res     = self._call_llm(payload)
            cible   = str(res.get("cible", "UNMATCHED")).strip()
            confiance = float(res.get("confiance", 0.0))
            justif  = str(res.get("justification", "")).strip()

            if cible not in known_uris:
                logger.warning(
                    f"  URI inconnue retournée par LLM : '{cible}'. "
                    f"Marqué UNMATCHED."
                )
                cible = "UNMATCHED"

            icon = "✅" if cible != "UNMATCHED" else "❌"
            logger.info(
                f"  {icon} LLM | '{col}' → {cible} (conf={confiance:.3f})"
            )
            return {
                "source_column": col,
                "target_uri":    cible,
                "confidence":    round(confiance, 4),
                "method":        "LLM_ONLY",
                "justification": justif,
            }

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                f"  ❌ Parsing JSON invalide pour '{col}'. "
                f"Erreur : {e}. Réponse brute : {res}"
            )
            return {
                "source_column": col,
                "target_uri":    "UNMATCHED",
                "confidence":    0.0,
                "method":        "LLM_ONLY",
                "justification": f"Erreur parsing LLM : {e}",
            }

    def run(self, source_path: str = "data/Source_B_RH.xlsx") -> dict:
        """
        Exécution complète : matching LLM-Only + évaluation + persistance.

        Returns:
            Dictionnaire complet des résultats, métriques et statistiques LLM.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("ABLATION B — LLM Only (LLaMA 3.1 / Groq)")
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

        # Matching colonne par colonne — TOUS les appels vont au LLM
        results = []
        for col in df.columns.tolist():
            logger.info(f"  → Envoi de '{col}' au LLM...")
            result = self._match_column(col, df)
            results.append(result)

        metrics  = compute_metrics(results, GOLD_STANDARD)
        elapsed  = round(time.time() - start, 2)

        # Statistiques latence LLM
        avg_latency_ms = (
            round(sum(self._call_times) / len(self._call_times), 1)
            if self._call_times else 0.0
        )
        total_llm_time_s = round(sum(self._call_times) / 1000, 2)

        output = {
            "variant":              "LLM_Only",
            "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s":            elapsed,
            "total_columns":        len(results),
            "llm_calls":            self._llm_calls,
            "avg_latency_per_call_ms": avg_latency_ms,
            "total_llm_time_s":     total_llm_time_s,
            "results":              results,
            "metrics":              metrics,
        }

        self.output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("-" * 60)
        logger.info(f"Précision : {metrics['precision']:.4f}")
        logger.info(f"Rappel    : {metrics['recall']:.4f}")
        logger.info(f"F1-score  : {metrics['f1']:.4f}")
        logger.info(
            f"Temps total : {elapsed}s  |  "
            f"Appels LLM : {self._llm_calls}  |  "
            f"Latence moy. : {avg_latency_ms}ms/appel  |  "
            f"Temps LLM cumulé : {total_llm_time_s}s"
        )
        logger.info(f"Résultats → '{self.output_path}'")
        logger.info("=" * 60)

        return output


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    matcher = LLMOnlyMatcher(
        ontology=ONTOLOGIE_CIBLE,
        llm_model="llama-3.1-8b-instant",
        temperature=0.0,
        output_path=Path("outputs/ablation_llm_only.json"),
    )
    matcher.run("data/Source_B_RH.xlsx")
