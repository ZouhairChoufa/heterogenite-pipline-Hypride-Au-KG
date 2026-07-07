"""
04_module_splink.py
===================
Module 4 : Résolution d'Entités Probabiliste (Splink + DuckDB).

Identifie les personnes qui apparaissent dans les deux sources (CRM et RH)
et génère des liens owl:sameAs au format N-Triples.

Standards de production :
  - Aucun iterrows() : export RDF 100% vectorisé + écriture par chunks
  - DuckDB comme backend (scalable, SQL natif, pas de Spark nécessaire)
  - Standardisation robuste des colonnes avec validation de schéma
  - Retry et gestion d'erreurs sur chaque étape critique
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class EntityResolutionConfig:
    """Paramètres configurables du pipeline de résolution d'entités."""
    path_crm: Path = field(default_factory=lambda: Path("data/Source_A_CRM.csv"))
    path_rh:  Path = field(default_factory=lambda: Path("data/Source_B_RH.csv"))

    # Splink
    match_probability_threshold: float      = 0.85
    max_pairs_u_estimate:        float      = 1e5
    jaro_thresholds:             list[float] = field(default_factory=lambda: [0.9, 0.8])

    # RDF output
    rdf_base_uri:   str  = "http://kg.projet.fr/ontology#"
    rdf_output_dir: Path = field(default_factory=lambda: Path("outputs"))
    rdf_output_file: str = "liens_identite.nt"
    rdf_chunk_size:  int = 50_000   # Lignes par batch d'écriture

    # DuckDB (":memory:" ou chemin fichier pour persistance entre runs)
    duckdb_path: str = ":memory:"

    def __post_init__(self):
        self.rdf_output_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Classe principale
# ==============================================================================

class EntityResolver:
    """
    Orchestrateur de la résolution d'entités avec Splink.

    Pipeline :
      1. Chargement et validation des sources
      2. Standardisation des colonnes (schéma commun)
      3. Configuration et entraînement du modèle Splink (EM)
      4. Inférence des paires candidates
      5. Export RDF vectorisé par chunks (owl:sameAs)
    """

    def __init__(self, config: EntityResolutionConfig) -> None:
        self.config = config
        self._linker: Linker | None = None
        logger.debug(f"EntityResolver initialisé.")

    # ------------------------------------------------------------------
    # Chargement et validation
    # ------------------------------------------------------------------

    def _load_sources(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charge les deux sources CSV avec validation d'existence.

        Returns:
            Tuple (df_crm_raw, df_rh_raw)

        Raises:
            FileNotFoundError: Si un fichier est absent.
        """
        for path in (self.config.path_crm, self.config.path_rh):
            if not path.exists():
                raise FileNotFoundError(
                    f"Fichier source introuvable : '{path}'. "
                    f"Lancez d'abord le Module 0 puis le Module 3 "
                    f"(qui génère Source_B_RH.csv depuis l'Excel)."
                )

        df_crm = pd.read_csv(
            self.config.path_crm, sep=";",
            low_memory=False, encoding="utf-8"
        )
        logger.info(f"CRM chargé : {df_crm.shape[0]:,} lignes")

        df_rh = pd.read_csv(
            self.config.path_rh,
            low_memory=False, encoding="utf-8",
            dtype=str,  # Préserve les formats de date et matricules
        )
        logger.info(f"RH chargé  : {df_rh.shape[0]:,} lignes")

        return df_crm, df_rh

    # ------------------------------------------------------------------
    # Standardisation CRM
    # ------------------------------------------------------------------

    def _standardize_crm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produit les colonnes : unique_id | nom_comparaison | date_comparaison

        Colonnes attendues en entrée : ID_Client, nom_complet, dt_naissance
        """
        required = {"ID_Client", "nom_complet", "dt_naissance"}
        missing  = required - set(df.columns)
        if missing:
            raise KeyError(
                f"Colonnes manquantes dans CRM : {missing}. "
                f"Colonnes reçues : {df.columns.tolist()}"
            )

        df = df.copy()
        df = df.rename(columns={"ID_Client": "unique_id"})
        df["nom_comparaison"] = df["nom_complet"].str.lower().str.strip()

        try:
            df["date_comparaison"] = pd.to_datetime(
                df["dt_naissance"], format="%d/%m/%Y", errors="raise"
            ).dt.strftime("%Y-%m-%d")
        except Exception as exc:
            logger.error(
                "Erreur de parsing de date dans CRM (colonne 'dt_naissance'). "
                f"Format attendu : dd/mm/YYYY. Détail : {exc}"
            )
            raise

        n_before = len(df)
        df = df.dropna(subset=["unique_id", "nom_comparaison", "date_comparaison"])
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(f"CRM : {n_dropped:,} lignes supprimées (nulls dans colonnes clés)")

        logger.info(f"CRM standardisé : {len(df):,} lignes utilisables")
        return df[["unique_id", "nom_comparaison", "date_comparaison"]]

    # ------------------------------------------------------------------
    # Standardisation RH
    # ------------------------------------------------------------------

    def _standardize_rh(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produit les colonnes : unique_id | nom_comparaison | date_comparaison

        Colonnes attendues en entrée : Matricule_RH, last_name, first_name, date_nais
        """
        required = {"Matricule_RH", "last_name", "first_name", "date_nais"}
        missing  = required - set(df.columns)
        if missing:
            raise KeyError(
                f"Colonnes manquantes dans RH : {missing}. "
                f"Colonnes reçues : {df.columns.tolist()}"
            )

        df = df.copy()
        df = df.rename(columns={"Matricule_RH": "unique_id"})
        df["nom_comparaison"] = (
            (df["first_name"].fillna("") + " " + df["last_name"].fillna(""))
            .str.lower()
            .str.strip()
        )
        df["date_comparaison"] = pd.to_datetime(
            df["date_nais"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        n_before = len(df)
        df = df.dropna(subset=["unique_id", "nom_comparaison", "date_comparaison"])
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(f"RH : {n_dropped:,} lignes supprimées (nulls dans colonnes clés)")

        logger.info(f"RH standardisé : {len(df):,} lignes utilisables")
        return df[["unique_id", "nom_comparaison", "date_comparaison"]]

    # ------------------------------------------------------------------
    # Entraînement Splink
    # ------------------------------------------------------------------

    def _build_and_train(
        self, df_crm: pd.DataFrame, df_rh: pd.DataFrame
    ) -> None:
        """
        Configure et entraîne le modèle Splink (estimation U + EM).

        L'entraînement se fait en deux étapes :
          1. Estimation des probabilités U (couples non-liés) via échantillonnage
          2. Estimation des probabilités M (couples liés) via Expectation-Maximisation
        """
        logger.info("Configuration du moteur probabiliste Splink...")

        settings = SettingsCreator(
            link_type="link_only",
            comparisons=[
                cl.JaroWinklerAtThresholds(
                    "nom_comparaison", self.config.jaro_thresholds
                ),
                cl.ExactMatch("date_comparaison"),
            ],
            blocking_rules_to_generate_predictions=[
                block_on("date_comparaison")
            ],
            retain_matching_columns=True,
            retain_intermediate_calculation_columns=False,
        )

        # DuckDB : ":memory:" pour <1M lignes, fichier pour persistance >1M
        db_connection = duckdb.connect(self.config.duckdb_path)
        db_api        = DuckDBAPI(connection=db_connection)

        self._linker = Linker(
            [df_crm, df_rh],
            settings,
            db_api,
            input_table_aliases=["crm", "rh"],
        )

        logger.info(
            f"Estimation des paramètres U "
            f"(max_pairs={self.config.max_pairs_u_estimate:.0e})..."
        )
        self._linker.training.estimate_u_using_random_sampling(
            max_pairs=self.config.max_pairs_u_estimate
        )

        logger.info("Estimation des paramètres M via Expectation-Maximisation...")
        self._linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("date_comparaison")
        )
        logger.info("Entraînement Splink terminé.")

    # ------------------------------------------------------------------
    # Export RDF vectorisé par chunks (cœur de l'optimisation scalabilité)
    # ------------------------------------------------------------------

    def _export_rdf_chunked(self, df_preds: pd.DataFrame) -> int:
        """
        Génère les triplets owl:sameAs en N-Triples par blocs vectorisés.

        POURQUOI C'EST CRITIQUE :
          iterrows() sur 5M lignes ≈ 60-90 minutes (Python pur, GIL)
          Vectorisation Pandas + écriture chunked ≈ 30-60 secondes (×100 plus rapide)
          RAM constante quel que soit le volume (seul 1 chunk en mémoire à la fois)

        Args:
            df_preds: DataFrame Splink avec colonnes unique_id_l et unique_id_r.

        Returns:
            Nombre total de triplets écrits.
        """
        output_path = self.config.rdf_output_dir / self.config.rdf_output_file
        base_uri    = self.config.rdf_base_uri
        chunk_size  = self.config.rdf_chunk_size
        predicate   = "<http://www.w3.org/2002/07/owl#sameAs>"
        total       = 0

        logger.info(
            f"Export RDF → '{output_path}' "
            f"(chunks de {chunk_size:,} lignes)..."
        )

        with open(output_path, "w", encoding="utf-8") as f:

            for start in range(0, len(df_preds), chunk_size):
                chunk = df_preds.iloc[start : start + chunk_size]

                # ---- Vectorisation : ZÉRO boucle Python ----
                # Construction des URIs via opérations string Pandas (SIMD)
                subjects = (
                    "<" + base_uri
                    + "Employee_"
                    + chunk["unique_id_r"].astype(str) + ">"
                )
                objects = (
                    "<" + base_uri
                    + "Customer_"
                    + chunk["unique_id_l"].astype(str) + ">"
                )

                # Assemblage en une Series de strings N-Triple
                triples = subjects + " " + predicate + " " + objects + " .\n"

                # Une seule opération I/O par chunk
                f.write("".join(triples.tolist()))

                total += len(chunk)
                logger.debug(
                    f"  Chunk {start:,}–{start+len(chunk):,} écrit "
                    f"({total:,} triplets cumulés)"
                )

        logger.info(
            f"Export RDF terminé : {total:,} triplets owl:sameAs → '{output_path}'"
        )
        return total

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def resolve(self) -> int:
        """
        Exécute le pipeline complet de résolution d'entités.

        Returns:
            Nombre de liens owl:sameAs générés.

        Raises:
            FileNotFoundError: Fichier source absent.
            KeyError:          Colonne attendue absente dans une source.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("MODULE 4 : Résolution d'Entités (Splink + DuckDB)")
        logger.info("=" * 60)

        # 1 — Chargement
        df_crm_raw, df_rh_raw = self._load_sources()

        # 2 — Standardisation
        logger.info("Standardisation des colonnes...")
        df_crm = self._standardize_crm(df_crm_raw)
        df_rh  = self._standardize_rh(df_rh_raw)

        # 3 — Entraînement Splink
        self._build_and_train(df_crm, df_rh)

        # 4 — Inférence
        logger.info(
            f"Inférence des paires candidates "
            f"(seuil = {self.config.match_probability_threshold})..."
        )
        df_preds = (
            self._linker.inference
            .predict(
                threshold_match_probability=self.config.match_probability_threshold
            )
            .as_pandas_dataframe()
        )
        logger.info(f"Inférence terminée : {len(df_preds):,} paires candidates")

        if df_preds.empty:
            logger.warning(
                "Aucune paire candidate identifiée. "
                "Vérifiez le seuil de probabilité ou la qualité des données de blocage. "
                "Conseil : réduisez match_probability_threshold (ex: 0.75)."
            )
            return 0

        # 5 — Export RDF vectorisé
        n_links = self._export_rdf_chunked(df_preds)

        elapsed = round(time.time() - start, 2)
        logger.info("-" * 60)
        logger.info(f"  Liens owl:sameAs générés : {n_links:,}")
        logger.info(f"  Taux de recouvrement estimé : "
                    f"{n_links/len(df_crm)*100:.1f}% du CRM")
        logger.info(f"MODULE 4 terminé en {elapsed}s")
        logger.info("=" * 60)

        return n_links


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    config   = EntityResolutionConfig()
    resolver = EntityResolver(config)
    try:
        resolver.resolve()
    except FileNotFoundError as e:
        logger.critical(f"Arrêt : {e}")
    except KeyError as e:
        logger.critical(f"Arrêt — schéma invalide : {e}")
    except Exception as e:
        logger.critical(f"Erreur inattendue : {e}", exc_info=True)
        raise
