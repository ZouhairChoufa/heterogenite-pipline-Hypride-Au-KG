"""
01_module_ingestion.py
======================
Module 1 : Ingestion et Profilage des données brutes.

Lit les deux sources (CSV + Excel), génère des rapports HTML ydata-profiling
et produit un résumé comparatif de la qualité des données.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class IngestionConfig:
    """Paramètres du module d'ingestion et de profilage."""
    path_crm:     Path = field(default_factory=lambda: Path("data/Source_A_CRM.csv"))
    path_rh:      Path = field(default_factory=lambda: Path("data/Source_B_RH.xlsx"))
    output_dir:   Path = field(default_factory=lambda: Path("reports"))
    minimal_mode: bool = True     # True = rapport rapide, False = rapport complet
    crm_sep:      str  = ";"


# ==============================================================================
# Classe principale
# ==============================================================================

class DataIngestionProfiler:
    """
    Charge, valide et profile les sources de données entrantes.

    Responsabilités :
      1. Validation d'existence des fichiers
      2. Chargement avec gestion d'encodage
      3. Calcul de statistiques qualité (taux de nullité, doublons)
      4. Génération de rapports HTML ydata-profiling
    """

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._df_crm: pd.DataFrame | None = None
        self._df_rh:  pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Chargement sécurisé
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path, sep: str = ";") -> pd.DataFrame:
        """Charge un CSV avec gestion des erreurs d'encodage."""
        if not path.exists():
            raise FileNotFoundError(f"Fichier CSV introuvable : '{path}'")
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)
            logger.info(f"Chargé '{path.name}' : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
            return df
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 échoué pour '{path.name}', tentative avec latin-1...")
            df = pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)
            logger.info(f"Chargé '{path.name}' (latin-1) : {df.shape}")
            return df

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """Charge un Excel avec validation."""
        if not path.exists():
            raise FileNotFoundError(f"Fichier Excel introuvable : '{path}'")
        df = pd.read_excel(path, engine="openpyxl")
        logger.info(f"Chargé '{path.name}' : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        return df

    # ------------------------------------------------------------------
    # Statistiques qualité
    # ------------------------------------------------------------------

    def _quality_report(self, df: pd.DataFrame, name: str) -> dict:
        """Calcule les métriques de qualité de données pour le log."""
        n_rows, n_cols = df.shape
        null_pct = (df.isnull().sum().sum() / (n_rows * n_cols) * 100)
        dup_rows  = df.duplicated().sum()

        report = {
            "source":          name,
            "rows":            n_rows,
            "columns":         n_cols,
            "null_pct":        round(null_pct, 2),
            "duplicate_rows":  int(dup_rows),
            "column_names":    df.columns.tolist(),
        }

        logger.info(
            f"[{name}] Qualité → {n_rows:,} lignes | {n_cols} cols | "
            f"{null_pct:.1f}% nulls | {dup_rows} doublons"
        )

        # Détail des colonnes à fort taux de nullité (> 20%)
        high_null = df.columns[df.isnull().mean() > 0.20].tolist()
        if high_null:
            logger.warning(f"[{name}] Colonnes avec > 20% de nulls : {high_null}")

        return report

    # ------------------------------------------------------------------
    # Profilage ydata-profiling
    # ------------------------------------------------------------------

    def _generate_profile(
        self, df: pd.DataFrame, title: str, output_file: str
    ) -> None:
        """Génère un rapport HTML ydata-profiling et le sauvegarde."""
        output_path = self.config.output_dir / output_file
        logger.info(f"Génération du rapport '{title}' → '{output_path}'...")

        profile = ProfileReport(
            df,
            title=title,
            minimal=self.config.minimal_mode,
            explorative=False,
        )
        profile.to_file(str(output_path))
        logger.info(f"Rapport sauvegardé : '{output_path}'")

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Exécute le pipeline complet d'ingestion et de profilage.

        Returns:
            Tuple (df_crm, df_rh) chargés et validés.

        Raises:
            FileNotFoundError: Si un fichier source est absent.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("MODULE 1 : Ingestion et Profilage")
        logger.info("=" * 60)

        # Chargement
        self._df_crm = self._load_csv(self.config.path_crm, sep=self.config.crm_sep)
        self._df_rh  = self._load_excel(self.config.path_rh)

        # Statistiques qualité
        self._quality_report(self._df_crm, "Source_A_CRM")
        self._quality_report(self._df_rh,  "Source_B_RH")

        # Affichage de l'hétérogénéité de schéma
        logger.info(
            f"Colonnes CRM : {self._df_crm.columns.tolist()}"
        )
        logger.info(
            f"Colonnes RH  : {self._df_rh.columns.tolist()}"
        )
        logger.info(
            "→ Hétérogénéité détectée (noms, formats, granularité) — "
            "schema matching nécessaire."
        )

        # Rapports HTML
        self._generate_profile(
            self._df_crm,
            title="Profilage Source A — CRM",
            output_file="profil_source_A.html",
        )
        self._generate_profile(
            self._df_rh,
            title="Profilage Source B — RH",
            output_file="profil_source_B.html",
        )

        elapsed = round(time.time() - start, 2)
        logger.info(f"MODULE 1 terminé en {elapsed}s")
        logger.info("=" * 60)

        return self._df_crm, self._df_rh


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    config = IngestionConfig()
    profiler = DataIngestionProfiler(config)
    profiler.run()
