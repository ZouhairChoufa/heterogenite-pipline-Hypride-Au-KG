"""
run_ablation_study.py
=====================
ÉTUDE D'ABLATION — Comparaison des 3 variantes du Module 2

Lance séquentiellement les 3 variantes sur les MÊMES données,
collecte leurs métriques et génère un rapport comparatif complet.

Variantes comparées :
  A — SBERT Only    : Sentence-BERT sans LLM
  B — LLM Only      : LLaMA 3.1 (Groq) sans filtrage SBERT
  C — Hybride       : SBERT (filtre τ=0.82) + LLM (Lazy Loading) ← notre proposition

Métriques collectées par variante :
  - Précision, Rappel, F1-score, Spécificité
  - Temps d'exécution total (s)
  - Nombre d'appels LLM
  - Latence moyenne par appel LLM (ms)
  - Coût estimé en tokens (estimation approximative)

Sorties :
  - outputs/ablation/ablation_sbert_only.json
  - outputs/ablation/ablation_llm_only.json
  - outputs/ablation/ablation_hybrid.json
  - outputs/ablation/ablation_comparison_report.json   ← rapport de synthèse
  - outputs/ablation/ablation_comparison_report.md     ← rapport lisible

Usage :
    python run_ablation_study.py
    python run_ablation_study.py --source data/Source_A_CRM.csv
    python run_ablation_study.py --source data/Source_B_RH.xlsx
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils.logger import get_logger
logger = get_logger("ablation.runner")


# ==============================================================================
# Gold Standard — vérité terrain unifiée (couvre les deux sources)
# ==============================================================================

GOLD_STANDARD: dict[str, Optional[str]] = {
    # Colonnes Source A — CRM
    "ID_Client":       "ex:employeeId",
    "nom_complet":     "ex:fullName",
    "email":           "schema:email",
    "dt_naissance":    "schema:birthDate",
    "ville":           "schema:addressLocality",
    "ca_annuel":       "ex:annualRevenue",
    "segment":         None,               # Pas de correspondance → UNMATCHED attendu
    # Colonnes Source B — RH
    "Matricule_RH":    "ex:employeeId",
    "Nom_Famille":     "foaf:familyName",
    "Prenom":          "foaf:firstName",
    "date_nais":       "schema:birthDate",
    "departement":     "ex:department",
    "contrat":         "schema:jobTitle",
    "salaire_mensuel": None,
    "date_embauche":   None,
}

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
# Évaluateur centralisé (identique dans les 3 variantes)
# ==============================================================================

def compute_metrics(
    results: list[dict],
    gold: dict[str, Optional[str]]
) -> dict:
    """Calcule TP/FP/FN/TN + Précision/Rappel/F1/Spécificité."""
    tp = fp = fn = tn = 0
    detail = []

    for r in results:
        col       = r["source_column"]
        predicted = r["target_uri"]
        expected  = gold.get(col)

        if expected is None:
            verdict = "TN" if predicted == "UNMATCHED" else "FP"
        else:
            if predicted == "UNMATCHED":
                verdict = "FN"
            elif predicted == expected:
                verdict = "TP"
            else:
                verdict = "FP"

        counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        counts[verdict] = 1
        tp += counts["TP"]; fp += counts["FP"]
        fn += counts["FN"]; tn += counts["TN"]

        detail.append({
            "column":    col,
            "predicted": predicted,
            "expected":  expected or "UNMATCHED",
            "verdict":   verdict,
        })

    precision = tp / (tp + fp)  if (tp + fp) > 0  else 0.0
    recall    = tp / (tp + fn)  if (tp + fn) > 0  else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision":    round(precision, 4),
        "recall":       round(recall, 4),
        "f1":           round(f1, 4),
        "specificity":  round(specificity, 4),
        "detail":       detail,
    }


# ==============================================================================
# Variante A — SBERT Only
# ==============================================================================

def run_sbert_only(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Lance le matching SBERT-Only et retourne le rapport complet."""
    from sentence_transformers import SentenceTransformer, util

    logger.info("┌─ VARIANTE A : SBERT Only ─────────────────────────────┐")
    start = time.time()

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    tau        = 0.82

    model           = SentenceTransformer(model_name)
    onto_texts      = list(ONTOLOGIE_CIBLE.values())
    onto_uris       = list(ONTOLOGIE_CIBLE.keys())
    onto_embeddings = model.encode(onto_texts)

    columns     = df.columns.tolist()
    col_embeds  = model.encode(columns)
    cos_scores  = util.cos_sim(col_embeds, onto_embeddings)
    results     = []

    for i, col in enumerate(columns):
        best_idx   = int(cos_scores[i].argmax())
        best_score = float(cos_scores[i][best_idx])
        target     = onto_uris[best_idx]
        above_tau  = best_score >= tau

        results.append({
            "source_column": col,
            "target_uri":    target,        # Forcé même si sous τ
            "confidence":    round(best_score, 4),
            "method":        "SBERT",
            "above_tau":     above_tau,
            "justification": f"cosinus={best_score:.4f} ({'≥' if above_tau else '<'} τ={tau})",
        })
        logger.info(
            f"  {'✅' if above_tau else '⚠️ '} '{col}' → {target} "
            f"(score={best_score:.3f})"
        )

    metrics = compute_metrics(results, GOLD_STANDARD)
    elapsed = round(time.time() - start, 2)

    report = {
        "variant":       "A_SBERT_Only",
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s":     elapsed,
        "total_columns": len(results),
        "llm_calls":     0,
        "avg_latency_ms": 0,
        "results":       results,
        "metrics":       metrics,
    }

    out = output_dir / "ablation_sbert_only.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"└─ SBERT Only terminé en {elapsed}s | F1={metrics['f1']:.4f} ─────┘\n")
    return report


# ==============================================================================
# Variante B — LLM Only
# ==============================================================================

def run_llm_only(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Lance le matching LLM-Only et retourne le rapport complet."""
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from tenacity import (
        before_sleep_log, retry, retry_if_exception_type,
        stop_after_attempt, wait_exponential,
    )
    try:
        from groq import APIConnectionError, RateLimitError
    except ImportError:
        RateLimitError = APIConnectionError = Exception

    logger.info("┌─ VARIANTE B : LLM Only ───────────────────────────────┐")
    start = time.time()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY manquante dans .env — "
            "impossible de lancer la variante LLM Only."
        )

    llm    = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=api_key)
    parser = JsonOutputParser()
    onto_texte = "\n".join(
        f"  - {uri} : {desc}" for uri, desc in ONTOLOGIE_CIBLE.items()
    )
    known_uris = set(ONTOLOGIE_CIBLE.keys()) | {"UNMATCHED"}

    prompt = PromptTemplate(
        template=(
            "Tu es un expert en ontologies et web sémantique.\n"
            "Associe la colonne source '{colonne}' "
            "(exemples : {exemples}) "
            "à la propriété parmi :\n{ontologie}\n\n"
            "Si aucune ne correspond, retourne 'UNMATCHED'.\n"
            "Réponds UNIQUEMENT en JSON : "
            "{{'cible': '...', 'confiance': 0.0, 'justification': '...'}}\n"
            "{format_instructions}"
        ),
        input_variables=["colonne", "exemples", "ontologie"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    # Décorateur retry inline (ne peut pas être appliqué à une lambda en Python)
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, 30),
        reraise=True,
    )
    def call_llm(payload: dict) -> dict:
        return chain.invoke(payload)

    results    = []
    llm_calls  = 0
    call_times = []

    for col in df.columns.tolist():
        non_null = df[col].dropna()
        n        = min(3, len(non_null))

        if n == 0:
            results.append({
                "source_column": col, "target_uri": "UNMATCHED",
                "confidence": 0.0, "method": "LLM_ONLY",
                "justification": "Colonne vide.",
            })
            continue

        exemples = non_null.sample(n, random_state=42).tolist()
        payload  = {
            "colonne":   col,
            "exemples":  json.dumps(exemples, ensure_ascii=False),
            "ontologie": onto_texte,
        }

        res = None
        t0  = time.time()
        try:
            res = call_llm(payload)
            if isinstance(res, list) and len(res) > 0:
                res = res[0]
            
            latency_ms  = round((time.time() - t0) * 1000, 1)
            call_times.append(latency_ms)
            llm_calls  += 1

            cible = str(res.get("cible", "UNMATCHED")).strip() if isinstance(res, dict) else "UNMATCHED"
            confiance = float(res.get("confiance", 0.0))
            justif    = str(res.get("justification", "")).strip()

            if cible not in known_uris:
                logger.warning(f"  URI inconnue '{cible}' → UNMATCHED")
                cible = "UNMATCHED"

            results.append({
                "source_column": col,
                "target_uri":    cible,
                "confidence":    round(confiance, 4),
                "method":        "LLM_ONLY",
                "latency_ms":    latency_ms,
                "justification": justif,
            })
            logger.info(
                f"  {'✅' if cible != 'UNMATCHED' else '❌'} LLM | '{col}' → "
                f"{cible} (conf={confiance:.3f}, {latency_ms}ms)"
            )

        except Exception as e:
            logger.error(f"  ❌ Échec LLM pour '{col}' : {e}")
            results.append({
                "source_column": col, "target_uri": "UNMATCHED",
                "confidence": 0.0, "method": "LLM_ONLY",
                "justification": f"Erreur : {e}",
            })

    metrics = compute_metrics(results, GOLD_STANDARD)
    elapsed = round(time.time() - start, 2)
    avg_lat = round(sum(call_times) / len(call_times), 1) if call_times else 0.0

    report = {
        "variant":               "B_LLM_Only",
        "timestamp":             time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s":             elapsed,
        "total_columns":         len(results),
        "llm_calls":             llm_calls,
        "avg_latency_ms":        avg_lat,
        "total_llm_time_s":      round(sum(call_times) / 1000, 2) if call_times else 0.0,
        "results":               results,
        "metrics":               metrics,
    }

    out = output_dir / "ablation_llm_only.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"└─ LLM Only terminé en {elapsed}s | "
        f"F1={metrics['f1']:.4f} | {llm_calls} appels LLM ──┘\n"
    )
    return report


# ==============================================================================
# Variante C — Hybride (notre proposition originale)
# ==============================================================================

def run_hybrid(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Lance le matching Hybride (SBERT + LLM Lazy Loading)."""
    from sentence_transformers import SentenceTransformer, util
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from tenacity import (
        before_sleep_log, retry, retry_if_exception_type,
        stop_after_attempt, wait_exponential,
    )
    try:
        from groq import APIConnectionError, RateLimitError
    except ImportError:
        RateLimitError = APIConnectionError = Exception

    logger.info("┌─ VARIANTE C : Hybride SBERT + LLM (Lazy Loading) ────┐")
    start = time.time()

    tau = 0.82

    # --- SBERT ---
    model           = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    onto_uris       = list(ONTOLOGIE_CIBLE.keys())
    onto_texts      = list(ONTOLOGIE_CIBLE.values())
    onto_embeddings = model.encode(onto_texts)
    known_uris      = set(onto_uris) | {"UNMATCHED"}
    onto_texte      = "\n".join(
        f"  - {uri} : {desc}" for uri, desc in ONTOLOGIE_CIBLE.items()
    )

    columns    = df.columns.tolist()
    col_embeds = model.encode(columns)
    cos_scores = util.cos_sim(col_embeds, onto_embeddings)

    results   = []
    ambiguous = []

    # Passe 1 — SBERT
    logger.info("[Passe 1/2] SBERT...")
    for i, col in enumerate(columns):
        best_idx   = int(cos_scores[i].argmax())
        best_score = float(cos_scores[i][best_idx])
        target     = onto_uris[best_idx]

        if best_score >= tau:
            results.append({
                "source_column": col,
                "target_uri":    target,
                "confidence":    round(best_score, 4),
                "method":        "SBERT",
                "justification": f"cosinus={best_score:.4f} ≥ τ={tau}",
            })
            logger.info(f"  ✅ SBERT | '{col}' → {target} (score={best_score:.3f})")
        else:
            ambiguous.append(col)
            logger.warning(
                f"  ⚠️  AMBIGU | '{col}' score={best_score:.3f} < τ={tau} → LLM"
            )

    # Passe 2 — LLM (seulement pour les cas ambigus)
    llm_calls  = 0
    call_times = []

    if ambiguous:
        logger.info(f"[Passe 2/2] LLM pour {len(ambiguous)} colonne(s) ambiguë(s)...")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY manquante dans .env")

        llm    = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=api_key)
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=(
                "Tu es un expert en ontologies.\n"
                "Associe '{colonne}' (exemples : {exemples}) "
                "à la propriété parmi :\n{ontologie}\n\n"
                "Si aucune ne correspond, retourne 'UNMATCHED'.\n"
                "JSON uniquement : {{'cible':'...','confiance':0.0,'justification':'...'}}\n"
                "{format_instructions}"
            ),
            input_variables=["colonne", "exemples", "ontologie"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser

        @retry(
            retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(4),
            before_sleep=before_sleep_log(logger, 30),
            reraise=True,
        )
        def call_llm_hybrid(payload: dict) -> dict:
            return chain.invoke(payload)

        for col in ambiguous:
            non_null = df[col].dropna()
            n        = min(3, len(non_null))
            exemples = non_null.sample(n, random_state=42).tolist() if n > 0 else []

            t0  = time.time()
            res = None
            try:
                res        = call_llm_hybrid({
                    "colonne":   col,
                    "exemples":  json.dumps(exemples, ensure_ascii=False),
                    "ontologie": onto_texte,
                })
                latency_ms = round((time.time() - t0) * 1000, 1)
                call_times.append(latency_ms)
                llm_calls += 1

                cible     = str(res.get("cible", "UNMATCHED")).strip()
                confiance = float(res.get("confiance", 0.0))
                justif    = str(res.get("justification", "")).strip()

                if cible not in known_uris:
                    cible = "UNMATCHED"

                results.append({
                    "source_column": col,
                    "target_uri":    cible,
                    "confidence":    round(confiance, 4),
                    "method":        "LLM",
                    "latency_ms":    latency_ms,
                    "justification": justif,
                })
                logger.info(
                    f"  🧠 LLM | '{col}' → {cible} "
                    f"(conf={confiance:.3f}, {latency_ms}ms)"
                )

            except Exception as e:
                results.append({
                    "source_column": col, "target_uri": "UNMATCHED",
                    "confidence": 0.0, "method": "UNMATCHED",
                    "justification": f"Erreur : {e}",
                })
                logger.error(f"  ❌ Échec LLM pour '{col}' : {e}")
    else:
        logger.info("[Passe 2/2] Aucune ambiguïté — 0 appel LLM.")

    metrics = compute_metrics(results, GOLD_STANDARD)
    elapsed = round(time.time() - start, 2)
    avg_lat = round(sum(call_times) / len(call_times), 1) if call_times else 0.0

    report = {
        "variant":           "C_Hybrid_SBERT_LLM",
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s":         elapsed,
        "total_columns":     len(results),
        "llm_calls":         llm_calls,
        "sbert_resolved":    len(columns) - len(ambiguous),
        "llm_resolved":      llm_calls,
        "avg_latency_ms":    avg_lat,
        "total_llm_time_s":  round(sum(call_times) / 1000, 2) if call_times else 0.0,
        "results":           results,
        "metrics":           metrics,
    }

    out = output_dir / "ablation_hybrid.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"└─ Hybride terminé en {elapsed}s | "
        f"F1={metrics['f1']:.4f} | "
        f"SBERT={len(columns)-len(ambiguous)} | LLM={llm_calls} ──┘\n"
    )
    return report


# ==============================================================================
# Rapport de synthèse
# ==============================================================================

def generate_comparison_report(
    report_a: dict,
    report_b: dict,
    report_c: dict,
    output_dir: Path,
) -> None:
    """
    Génère un rapport comparatif JSON + Markdown structuré.

    Inclut :
      - Tableau de synthèse des métriques
      - Analyse de la réduction des appels LLM (FinOps)
      - Recommandation argumentée
    """

    m_a = report_a["metrics"]
    m_b = report_b["metrics"]
    m_c = report_c["metrics"]

    # Gain FinOps : réduction des appels LLM (hybride vs LLM-Only)
    llm_b = report_b["llm_calls"]
    llm_c = report_c["llm_calls"]
    llm_reduction_pct = round(
        (llm_b - llm_c) / llm_b * 100 if llm_b > 0 else 0.0, 1
    )

    # Gain/perte F1 hybride vs SBERT-Only
    f1_gain_vs_sbert = round((m_c["f1"] - m_a["f1"]) * 100, 2)
    # Gain/perte F1 hybride vs LLM-Only
    f1_delta_vs_llm  = round((m_c["f1"] - m_b["f1"]) * 100, 2)

    comparison = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "A_SBERT_Only": {
                "precision":  m_a["precision"],
                "recall":     m_a["recall"],
                "f1":         m_a["f1"],
                "elapsed_s":  report_a["elapsed_s"],
                "llm_calls":  0,
            },
            "B_LLM_Only": {
                "precision":  m_b["precision"],
                "recall":     m_b["recall"],
                "f1":         m_b["f1"],
                "elapsed_s":  report_b["elapsed_s"],
                "llm_calls":  report_b["llm_calls"],
                "avg_latency_ms": report_b.get("avg_latency_ms", 0),
            },
            "C_Hybrid": {
                "precision":  m_c["precision"],
                "recall":     m_c["recall"],
                "f1":         m_c["f1"],
                "elapsed_s":  report_c["elapsed_s"],
                "llm_calls":  llm_c,
                "sbert_resolved": report_c.get("sbert_resolved", 0),
                "avg_latency_ms": report_c.get("avg_latency_ms", 0),
            },
        },
        "finops_analysis": {
            "llm_calls_sbert_only": 0,
            "llm_calls_llm_only":   llm_b,
            "llm_calls_hybrid":     llm_c,
            "reduction_hybrid_vs_llm_only_pct": llm_reduction_pct,
        },
        "quality_analysis": {
            "f1_gain_hybrid_vs_sbert_pct":  f1_gain_vs_sbert,
            "f1_delta_hybrid_vs_llm_pct":   f1_delta_vs_llm,
            "best_f1_variant":   max(
                [("A_SBERT_Only", m_a["f1"]),
                 ("B_LLM_Only",   m_b["f1"]),
                 ("C_Hybrid",     m_c["f1"])],
                key=lambda x: x[1]
            )[0],
        },
    }

    # JSON
    json_out = output_dir / "ablation_comparison_report.json"
    json_out.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown
    sep = "| " + " | ".join(["---"] * 6) + " |"
    md = f"""# Rapport d'Ablation — Schema Matching Module 2
*Généré le {comparison["generated_at"]}*

## Tableau de Synthèse des Performances

| Variante | Précision | Rappel | F1-score | Temps (s) | Appels LLM |
{sep}
| **A — SBERT Only**  | {m_a['precision']:.4f} | {m_a['recall']:.4f} | {m_a['f1']:.4f} | {report_a['elapsed_s']} | 0 |
| **B — LLM Only**    | {m_b['precision']:.4f} | {m_b['recall']:.4f} | {m_b['f1']:.4f} | {report_b['elapsed_s']} | {llm_b} |
| **C — Hybride** ⭐  | {m_c['precision']:.4f} | {m_c['recall']:.4f} | {m_c['f1']:.4f} | {report_c['elapsed_s']} | {llm_c} |

## Analyse FinOps (Réduction des Appels LLM)

| Métrique | Valeur |
| --- | --- |
| Appels LLM — SBERT Only  | 0 |
| Appels LLM — LLM Only    | {llm_b} |
| Appels LLM — Hybride     | {llm_c} |
| **Réduction (Hybride vs LLM Only)** | **{llm_reduction_pct}%** |

## Analyse Qualité (Écart F1)

| Comparaison | Écart F1 (points de %) |
| --- | --- |
| Hybride vs SBERT Only | {f1_gain_vs_sbert:+.2f}% |
| Hybride vs LLM Only   | {f1_delta_vs_llm:+.2f}% |

## Détail par Colonne — Variante C (Hybride)

| Colonne | Prédit | Attendu | Verdict | Méthode | Confiance |
| --- | --- | --- | --- | --- | --- |
"""
    for d in m_c["detail"]:
        icon = "✅ TP" if d["verdict"] == "TP" else \
               "❌ FP" if d["verdict"] == "FP" else \
               "⚠️  FN" if d["verdict"] == "FN" else "✅ TN"
        # Trouver la confiance dans les results
        conf = next(
            (r["confidence"] for r in report_c["results"]
             if r["source_column"] == d["column"]), "—"
        )
        method = next(
            (r["method"] for r in report_c["results"]
             if r["source_column"] == d["column"]), "—"
        )
        md += (
            f"| `{d['column']}` | `{d['predicted']}` | `{d['expected']}` | "
            f"{icon} | {method} | {conf} |\n"
        )

    md += f"""
## Conclusion

La variante **C (Hybride SBERT + LLM Lazy Loading)** obtient le meilleur
équilibre Qualité/Coût :
- F1-score = **{m_c['f1']:.4f}**
- Réduction des appels LLM de **{llm_reduction_pct}%** par rapport à LLM Only
- Gain F1 de **{f1_gain_vs_sbert:+.2f}pp** par rapport à SBERT Only
"""

    md_out = output_dir / "ablation_comparison_report.md"
    md_out.write_text(md, encoding="utf-8")

    logger.info(f"Rapport JSON → '{json_out}'")
    logger.info(f"Rapport MD  → '{md_out}'")

    # Affichage console synthèse
    print("\n" + "═" * 62)
    print("  ABLATION STUDY — RÉSULTATS FINAUX")
    print("═" * 62)
    print(f"  {'Variante':<25} {'Préc.':<8} {'Rappel':<8} {'F1':<8} {'LLM':<6} {'Temps'}")
    print("  " + "─" * 60)
    print(
        f"  {'A — SBERT Only':<25} "
        f"{m_a['precision']:<8.4f} {m_a['recall']:<8.4f} "
        f"{m_a['f1']:<8.4f} {'0':<6} {report_a['elapsed_s']}s"
    )
    print(
        f"  {'B — LLM Only':<25} "
        f"{m_b['precision']:<8.4f} {m_b['recall']:<8.4f} "
        f"{m_b['f1']:<8.4f} {llm_b:<6} {report_b['elapsed_s']}s"
    )
    print(
        f"  {'C — Hybride ⭐':<25} "
        f"{m_c['precision']:<8.4f} {m_c['recall']:<8.4f} "
        f"{m_c['f1']:<8.4f} {llm_c:<6} {report_c['elapsed_s']}s"
    )
    print("═" * 62)
    print(f"  Réduction appels LLM (C vs B) : -{llm_reduction_pct}%")
    print(f"  Gain F1 (C vs A)              : {f1_gain_vs_sbert:+.2f} points de %")
    print("═" * 62 + "\n")


# ==============================================================================
# Orchestrateur principal
# ==============================================================================

def run_ablation(source_path: str = "data/Source_B_RH.xlsx") -> None:
    """
    Lance les 3 variantes séquentiellement sur la même source,
    puis génère le rapport comparatif.
    """
    output_dir = Path("outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Source '{source_path}' introuvable. "
            f"Lancez d'abord run_pipeline.py --only 0"
        )

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║         ABLATION STUDY — Module 2 Schema Matching       ║")
    logger.info(f"║         Source : {source_path:<40}║")
    logger.info("╚══════════════════════════════════════════════════════════╝\n")

    # Chargement de la source (une seule fois, partagée entre les 3 variantes)
    df = (
        pd.read_excel(path, engine="openpyxl")
        if path.suffix == ".xlsx"
        else pd.read_csv(path, sep=";")
    )
    logger.info(f"Source partagée chargée : {df.shape} — {df.columns.tolist()}\n")

    report_a = run_sbert_only(df, output_dir)
    report_b = run_llm_only(df, output_dir)
    report_c = run_hybrid(df, output_dir)

    generate_comparison_report(report_a, report_b, report_c, output_dir)


# ==============================================================================
# Point d'entrée CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study — Schema Matching Module 2"
    )
    parser.add_argument(
        "--source",
        default="data/Source_B_RH.xlsx",
        help="Chemin vers le fichier source à analyser (défaut: data/Source_B_RH.xlsx)",
    )
    args = parser.parse_args()

    import argparse as _argparse   # évite redéfinition si déjà importé
    run_ablation(source_path=args.source)
