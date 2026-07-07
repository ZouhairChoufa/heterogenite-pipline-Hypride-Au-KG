"""
02_module_matching.py
=====================
Module 2 : Schema Matching Hybride (SBERT local + LLaMA 3.1 via Groq).

Stratégie en deux passes :
  Passe 1 — SBERT (local, rapide)  : résout les cas non-ambigus (score ≥ τ)
  Passe 2 — LLaMA 3.1 (Groq, LLM) : résout les cas ambigus restants

Sortie : matching_results.json (persisté pour audit et reprise)
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# Groq exceptions (importation défensive)
try:
    from groq import APIConnectionError, RateLimitError
except ImportError:
    # Fallback si groq n'est pas encore installé (évite un crash à l'import)
    RateLimitError = Exception
    APIConnectionError = Exception


# ==============================================================================
# Structures de données
# ==============================================================================

@dataclass
class OntologyTarget:
    """Propriété cible dans l'ontologie."""
    uri: str
    description: str


@dataclass
class MatchingResult:
    """Résultat du matching pour une colonne source."""
    source_column: str
    target_uri:    str
    confidence:    float
    method:        str          # "SBERT" | "LLM" | "UNMATCHED"
    justification: str = ""

    def to_dict(self) -> dict:
        return {
            "source_column": self.source_column,
            "target_uri":    self.target_uri,
            "confidence":    round(self.confidence, 4),
            "method":        self.method,
            "justification": self.justification,
        }


@dataclass
class SchemaMatcherConfig:
    """Paramètres configurables du Schema Matcher."""
    ontology: dict[str, str]
    sbert_model:  str   = "paraphrase-multilingual-MiniLM-L12-v2"
    sbert_tau:    float = 0.82
    llm_model:    str   = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0
    max_llm_retries: int   = 4
    n_value_samples: int   = 3
    output_path: Path = field(default_factory=lambda: Path("outputs/matching_results.json"))


# ==============================================================================
# Classe principale
# ==============================================================================

class SchemaMatcher:
    """
    Orchestrateur du Schema Matching hybride.

    Architecture :
      - Passe 1 (SBERT)  : embedding cosinus sur descriptions de l'ontologie
      - Passe 2 (LLM)    : appel Groq avec retry exponentiel pour les cas ambigus
      - Persistance JSON : résultats sauvegardés pour audit
    """

    def __init__(self, config: SchemaMatcherConfig) -> None:
        self.config = config
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.ontology_targets: list[OntologyTarget] = [
            OntologyTarget(uri=k, description=v)
            for k, v in config.ontology.items()
        ]
        self.results: list[MatchingResult] = []
        self._llm_chain = None   # Lazy init

        # Chargement SBERT + pré-encodage ontologie
        logger.info(f"Chargement du modèle SBERT : '{config.sbert_model}'...")
        self._sbert = SentenceTransformer(config.sbert_model)
        self._onto_embeddings = self._sbert.encode(
            [t.description for t in self.ontology_targets]
        )
        logger.info(
            f"Ontologie encodée : {len(self.ontology_targets)} propriétés cibles."
        )

    # ------------------------------------------------------------------
    # Point d'entrée public
    # ------------------------------------------------------------------

    def match(self, df: pd.DataFrame) -> list[MatchingResult]:
        """
        Lance le pipeline de matching sur toutes les colonnes de df.

        Args:
            df: DataFrame source dont on veut aligner les colonnes à l'ontologie.

        Returns:
            Liste de MatchingResult (une entrée par colonne).
        """
        columns = df.columns.tolist()
        logger.info(f"Démarrage du matching sur {len(columns)} colonnes : {columns}")

        ambiguous: list[str] = []

        # ---- Passe 1 : SBERT ----
        logger.info("[Passe 1/2] Analyse sémantique locale (SBERT)...")
        col_embeddings = self._sbert.encode(columns)
        cos_scores     = util.cos_sim(col_embeddings, self._onto_embeddings)

        for i, col in enumerate(columns):
            best_idx   = int(cos_scores[i].argmax())
            best_score = float(cos_scores[i][best_idx])

            if best_score >= self.config.sbert_tau:
                result = MatchingResult(
                    source_column=col,
                    target_uri=self.ontology_targets[best_idx].uri,
                    confidence=best_score,
                    method="SBERT",
                    justification=(
                        f"Similarité cosinus = {best_score:.4f} "
                        f"≥ seuil τ = {self.config.sbert_tau}"
                    ),
                )
                self.results.append(result)
                logger.info(
                    f"  ✅ SBERT | '{col}' → {result.target_uri} "
                    f"(conf={best_score:.3f})"
                )
            else:
                ambiguous.append(col)
                logger.warning(
                    f"  ⚠️  AMBIGU | '{col}' score={best_score:.3f} "
                    f"< τ={self.config.sbert_tau} → délégué au LLM"
                )

        # ---- Passe 2 : LLM ----
        if ambiguous:
            logger.info(
                f"[Passe 2/2] {len(ambiguous)} colonne(s) envoyée(s) "
                f"à LLaMA 3.1 via Groq..."
            )
            self._init_llm_chain()
            for col in ambiguous:
                result = self._match_with_llm(col, df)
                if result:
                    self.results.append(result)
                else:
                    # Fallback : colonne non matchée (tracée)
                    self.results.append(
                        MatchingResult(
                            source_column=col,
                            target_uri="UNMATCHED",
                            confidence=0.0,
                            method="UNMATCHED",
                            justification="Échec SBERT + LLM. Vérification manuelle requise.",
                        )
                    )
                    logger.error(
                        f"  ❌ '{col}' non mappée. Marquée UNMATCHED."
                    )
        else:
            logger.info("[Passe 2/2] Aucun cas ambigu — appel LLM non nécessaire.")

        self._save_results()
        return self.results

    # ------------------------------------------------------------------
    # Initialisation paresseuse du client LLM
    # ------------------------------------------------------------------

    def _init_llm_chain(self) -> None:
        """Instancie le client Groq uniquement si nécessaire (lazy loading)."""
        if self._llm_chain is not None:
            return

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Variable 'GROQ_API_KEY' non définie. "
                "Créez un fichier .env ou exécutez : export GROQ_API_KEY='gsk_...' "
                "Obtenez votre clé sur https://console.groq.com"
            )

        llm    = ChatGroq(
            model_name=self.config.llm_model,
            temperature=self.config.llm_temperature,
            api_key=api_key,
        )
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=(
                "Tu es un expert en ontologies et web sémantique.\n"
                "Associe la colonne source '{colonne}' "
                "(exemples de valeurs réelles : {exemples}) "
                "à la propriété ontologique la plus appropriée parmi :\n"
                "{ontologie}\n\n"
                "Renvoie UNIQUEMENT un objet JSON valide (sans markdown) "
                "avec exactement ces clés :\n"
                "  'cible'        : URI exact de la propriété choisie\n"
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
        logger.debug(f"Client LLM Groq initialisé (modèle='{self.config.llm_model}').")

    # ------------------------------------------------------------------
    # Appel LLM avec retry exponentiel
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, 30),  # 30 = WARNING
        reraise=True,
    )
    def _call_llm(self, payload: dict) -> dict:
        """Appel LLM protégé par retry exponentiel (rate limit & réseau)."""
        return self._llm_chain.invoke(payload)

    def _match_with_llm(
        self, col: str, df: pd.DataFrame
    ) -> Optional[MatchingResult]:
        """
        Mappe une colonne ambiguë via LLaMA 3.1.

        Gère :
          - Colonnes quasi-vides (< 1 valeur non-nulle)
          - Réponse JSON invalide ou incomplète du LLM
          - Erreurs réseau / rate limits (via retry)
        """
        non_null = df[col].dropna()
        n        = min(self.config.n_value_samples, len(non_null))

        if n == 0:
            logger.warning(f"  Colonne '{col}' : aucune valeur disponible. Skip LLM.")
            return None

        exemples    = non_null.sample(n, random_state=42).tolist()
        onto_texte  = "\n".join(
            f"  - {t.uri} : {t.description}" for t in self.ontology_targets
        )

        payload = {
            "colonne":  col,
            "exemples": json.dumps(exemples, ensure_ascii=False),
            "ontologie": onto_texte,
        }

        res = None
        try:
            res = self._call_llm(payload)

            # Guard clause : validation de la structure JSON retournée
            required_keys = {"cible", "confiance", "justification"}
            missing = required_keys - set(res.keys())
            if missing:
                raise ValueError(
                    f"Réponse LLM incomplète, clés manquantes : {missing}. "
                    f"Réponse brute : {res}"
                )

            # Validation de types
            cible        = str(res["cible"]).strip()
            confiance    = float(res["confiance"])
            justification = str(res["justification"]).strip()

            # Vérification que l'URI renvoyée existe dans l'ontologie
            known_uris = {t.uri for t in self.ontology_targets}
            if cible not in known_uris:
                logger.warning(
                    f"  LLM a renvoyé un URI inconnu : '{cible}'. "
                    f"URIs valides : {known_uris}. Marqué UNMATCHED."
                )
                return None

            result = MatchingResult(
                source_column=col,
                target_uri=cible,
                confidence=confiance,
                method="LLM",
                justification=justification,
            )
            logger.info(
                f"  🧠 LLM | '{col}' → {result.target_uri} "
                f"(conf={confiance:.3f}) — {justification}"
            )
            return result

        except (RateLimitError, APIConnectionError) as e:
            logger.error(
                f"  ❌ LLM : Échec définitif pour '{col}' (4 tentatives épuisées). "
                f"{type(e).__name__}: {e}"
            )
            return None

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                f"  ❌ LLM : Parsing JSON invalide pour '{col}'. "
                f"Erreur : {e}. Réponse brute : {res}"
            )
            return None

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        """Sérialise les résultats en JSON pour audit et reprise de pipeline."""
        output = {
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_columns": len(self.results),
            "sbert_count":   sum(1 for r in self.results if r.method == "SBERT"),
            "llm_count":     sum(1 for r in self.results if r.method == "LLM"),
            "unmatched":     sum(1 for r in self.results if r.method == "UNMATCHED"),
            "results":       [r.to_dict() for r in self.results],
        }
        self.config.output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Résultats persistés → '{self.config.output_path}'")

    def get_dataframe(self) -> pd.DataFrame:
        """Retourne les résultats sous forme de DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])


# ==============================================================================
# Ontologie cible (partagée entre les modules)
# ==============================================================================

ONTOLOGIE_CIBLE: dict[str, str] = {
    "ex:employeeId":   "Identifiant unique de l'employé ou du client",
    "schema:birthDate":"Date de naissance de la personne",
    "foaf:familyName": "Nom de famille de la personne",
    "foaf:firstName":  "Prénom de la personne",
    "ex:fullName":     "Nom complet (prénom + nom de famille concaténés)",
    "schema:email":    "Adresse email de contact",
    "ex:department":   "Département ou service dans l'organisation",
    "ex:annualRevenue":"Chiffre d'affaires ou revenu annuel",
    "schema:jobTitle": "Poste ou type de contrat professionnel",
    "schema:addressLocality": "Ville ou localité",
}


# ==============================================================================
# Point d'entrée
# ==============================================================================

def run_schema_matching(source_path: str = "data/Source_B_RH.xlsx") -> list[MatchingResult]:
    """
    Fonction d'entrée principale du Module 2.

    Args:
        source_path: Chemin vers le fichier source (Excel ou CSV).

    Returns:
        Liste des MatchingResult.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("MODULE 2 : Schema Matching Hybride")
    logger.info("=" * 60)

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : '{source_path}'")

    # Chargement
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=";", low_memory=False)
    else:
        raise ValueError(f"Format non supporté : '{path.suffix}'")

    logger.info(f"Source chargée : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    config  = SchemaMatcherConfig(
        ontology=ONTOLOGIE_CIBLE,
        sbert_tau=0.82,
        output_path=Path("outputs/matching_results.json"),
    )
    matcher = SchemaMatcher(config)
    results = matcher.match(df)

    elapsed = round(time.time() - start, 2)
    df_res  = matcher.get_dataframe()

    logger.info("-" * 60)
    logger.info(f"MODULE 2 terminé en {elapsed}s")
    logger.info(f"  Total colonnes : {len(results)}")
    logger.info(f"  SBERT  : {(df_res['method']=='SBERT').sum()}")
    logger.info(f"  LLM    : {(df_res['method']=='LLM').sum()}")
    logger.info(f"  UNMATCHED : {(df_res['method']=='UNMATCHED').sum()}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_schema_matching("data/Source_B_RH.xlsx")
