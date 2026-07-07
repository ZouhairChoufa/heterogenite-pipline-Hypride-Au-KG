"""
fix_variante_b.py
=================
Variante B (LLM Only) — VERSION CORRIGÉE
Fix du bug : 'list' object has no attribute 'get'

Usage :
    python fix_variante_b.py
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# ==============================================================================
# Ontologie cible
# ==============================================================================

ONTOLOGIE = {
    "ex:employeeId":          "Identifiant unique de l employé ou du client",
    "schema:birthDate":       "Date de naissance de la personne",
    "foaf:familyName":        "Nom de famille de la personne",
    "foaf:firstName":         "Prénom de la personne",
    "ex:fullName":            "Nom complet (prénom + nom de famille concaténés)",
    "schema:email":           "Adresse email de contact",
    "ex:department":          "Département ou service dans l organisation",
    "ex:annualRevenue":       "Chiffre d affaires ou revenu annuel",
    "schema:jobTitle":        "Poste ou type de contrat professionnel",
    "schema:addressLocality": "Ville ou localité",
}

# ==============================================================================
# Gold Standard — Source B (RH)
# ==============================================================================

GOLD = {
    "Matricule_RH":    "ex:employeeId",
    "Nom_Famille":     "foaf:familyName",
    "Prenom":          "foaf:firstName",
    "date_nais":       "schema:birthDate",
    "departement":     "ex:department",
    "contrat":         "schema:jobTitle",
    "salaire_mensuel": None,       # Pas de correspondance → UNMATCHED attendu
    "date_embauche":   None,       # Pas de correspondance → UNMATCHED attendu
}

# ==============================================================================
# Configuration LLM
# ==============================================================================

def build_chain():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY manquante dans .env\n"
            "Ajoutez : GROQ_API_KEY=gsk_xxxx dans votre fichier .env"
        )

    llm    = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=api_key)
    parser = JsonOutputParser()

    onto_texte = "\n".join(
        f"  - {uri} : {desc}" for uri, desc in ONTOLOGIE.items()
    )

    prompt = PromptTemplate(
        template=(
            "Tu es un expert en ontologies et web sémantique.\n"
            "Associe la colonne source '{colonne}' "
            "(exemples de valeurs : {exemples}) "
            "a la propriete ontologique parmi :\n"
            "{ontologie}\n\n"
            "REGLE IMPORTANTE : Si aucune propriete ne correspond, "
            "retourne exactement UNMATCHED pour la cle cible.\n\n"
            "Reponds UNIQUEMENT avec un objet JSON valide, "
            "sans markdown, sans explication, sans code :\n"
            "{{\"cible\": \"URI_ou_UNMATCHED\", "
            "\"confiance\": 0.95, "
            "\"justification\": \"raison en moins de 10 mots\"}}\n\n"
            "{format_instructions}"
        ),
        input_variables=["colonne", "exemples", "ontologie"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    chain = prompt | llm | parser
    return chain, onto_texte


# ==============================================================================
# Parsing sécurisé — FIX du bug list/dict
# ==============================================================================

def safe_parse(res) -> dict:
    """
    Gère les deux formats de réponse possibles de LLaMA 3.1 :
      - Format dict  : {"cible": "...", "confiance": 0.9, ...}   → correct
      - Format list  : [{"cible": "...", ...}]                    → bug LangChain
    """
    if isinstance(res, list):
        if len(res) > 0 and isinstance(res[0], dict):
            return res[0]
        raise ValueError(f"Liste vide ou format inattendu : {res}")

    if isinstance(res, dict):
        return res

    raise ValueError(f"Type inattendu : {type(res)} — valeur : {res}")


# ==============================================================================
# Calcul des métriques
# ==============================================================================

def compute_metrics(results: list[dict], gold: dict) -> dict:
    tp = fp = fn = tn = 0
    detail = []

    for r in results:
        col      = r["source_column"]
        pred     = r["target_uri"]
        expected = gold.get(col)

        if expected is None:
            verdict = "TN" if pred == "UNMATCHED" else "FP"
        else:
            if pred == "UNMATCHED":
                verdict = "FN"
            elif pred == expected:
                verdict = "TP"
            else:
                verdict = "FP"

        if verdict == "TP": tp += 1
        elif verdict == "FP": fp += 1
        elif verdict == "FN": fn += 1
        else: tn += 1

        detail.append({
            "column":    col,
            "predicted": pred,
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
# Main
# ==============================================================================

def main():
    print("=" * 50)
    print("  VARIANTE B (CORRIGÉE) — LLM Only")
    print("  Source : data/Source_B_RH.xlsx")
    print("=" * 50)

    # Chargement de la source
    source_path = Path("data/Source_B_RH.xlsx")
    if not source_path.exists():
        raise FileNotFoundError(
            "data/Source_B_RH.xlsx introuvable.\n"
            "Lancez d abord : python run_pipeline.py --only 0"
        )

    df = pd.read_excel(source_path, engine="openpyxl")
    print(f"\nSource chargée : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"Colonnes : {df.columns.tolist()}\n")

    # Construction de la chaîne LLM
    chain, onto_texte = build_chain()
    known_uris = set(ONTOLOGIE.keys()) | {"UNMATCHED"}

    results    = []
    llm_calls  = 0
    call_times = []
    start_total = time.time()

    # Matching colonne par colonne
    for col in df.columns.tolist():
        non_null = df[col].dropna()
        n        = min(3, len(non_null))
        exemples = non_null.sample(n, random_state=42).tolist() if n > 0 else []

        payload = {
            "colonne":   col,
            "exemples":  json.dumps(exemples, ensure_ascii=False),
            "ontologie": onto_texte,
        }

        t0  = time.time()
        res = None

        try:
            raw_res    = chain.invoke(payload)
            latency_ms = round((time.time() - t0) * 1000, 1)
            call_times.append(latency_ms)
            llm_calls += 1

            # --- FIX PRINCIPAL ---
            res       = safe_parse(raw_res)
            cible     = str(res.get("cible", "UNMATCHED")).strip()
            confiance = float(res.get("confiance", 0.0))
            justif    = str(res.get("justification", "")).strip()

            # Validation URI
            if cible not in known_uris:
                print(f"  ⚠️  URI inconnue '{cible}' → forcé UNMATCHED")
                cible = "UNMATCHED"

            results.append({
                "source_column": col,
                "target_uri":    cible,
                "confidence":    round(confiance, 4),
                "method":        "LLM_ONLY",
                "latency_ms":    latency_ms,
                "justification": justif,
            })

            icon = "✅" if cible != "UNMATCHED" else "❌"
            print(
                f"  {icon} LLM | '{col}' → {cible} "
                f"(conf={confiance:.3f}, {latency_ms}ms)"
            )

        except Exception as e:
            latency_ms = round((time.time() - t0) * 1000, 1)
            results.append({
                "source_column": col,
                "target_uri":    "UNMATCHED",
                "confidence":    0.0,
                "method":        "LLM_ONLY",
                "latency_ms":    latency_ms,
                "justification": f"Erreur : {str(e)[:100]}",
            })
            print(f"  ❌ ERREUR '{col}' : {str(e)[:80]}")

    # Métriques
    elapsed    = round(time.time() - start_total, 2)
    metrics    = compute_metrics(results, GOLD)
    avg_lat    = round(sum(call_times) / len(call_times), 1) if call_times else 0.0
    total_llm  = round(sum(call_times) / 1000, 2)

    # Affichage résultats
    print()
    print("=" * 50)
    print("  RÉSULTATS FINAUX — VARIANTE B (CORRIGÉE)")
    print("=" * 50)
    print(f"  Précision    : {metrics['precision']:.4f}")
    print(f"  Rappel       : {metrics['recall']:.4f}")
    print(f"  F1-score     : {metrics['f1']:.4f}")
    print(f"  Spécificité  : {metrics['specificity']:.4f}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"  Appels LLM   : {llm_calls}")
    print(f"  Latence moy  : {avg_lat} ms/appel")
    print(f"  Temps LLM    : {total_llm}s")
    print(f"  Temps total  : {elapsed}s")
    print("=" * 50)

    # Détail par colonne
    print("\n  DÉTAIL PAR COLONNE :")
    print(f"  {'Colonne':<20} {'Prédit':<25} {'Attendu':<25} {'Verdict'}")
    print("  " + "-" * 80)
    for d in metrics["detail"]:
        icon = "✅" if d["verdict"] == "TP" else \
               "✅" if d["verdict"] == "TN" else \
               "❌" if d["verdict"] == "FP" else "⚠️ "
        print(
            f"  {d['column']:<20} {d['predicted']:<25} "
            f"{d['expected']:<25} {icon} {d['verdict']}"
        )

    # Sauvegarde
    output_dir = Path("outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_llm_only_FIXED.json"

    report = {
        "variant":           "B_LLM_Only_FIXED",
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s":         elapsed,
        "total_columns":     len(results),
        "llm_calls":         llm_calls,
        "avg_latency_ms":    avg_lat,
        "total_llm_time_s":  total_llm,
        "results":           results,
        "metrics":           metrics,
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n  Résultats sauvegardés → {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
