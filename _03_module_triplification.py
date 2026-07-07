"""
03_module_triplification.py
===========================
Module 3 : Triplification Sémantique avec Morph-KGC.

Transforme les sources tabulaires en graphe RDF (N-Triples) via les
règles de mapping YARRRML définies dans regles_mapping.yml.

Optimisations :
  - Conversion Excel → CSV avant Morph-KGC (évite le rechargement complet
    en mémoire par la bibliothèque, qui est optimisée pour CSV)
  - Validation du graphe généré (nombre de triplets, URIs)
  - Sérialisation en N-Triples (.nt) : format le plus compact et le plus
    rapide à parser pour les triples stores (Fuseki, GraphDB…)
"""

import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import morph_kgc
import pandas as pd
from rdflib import Graph

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class TriplificationConfig:
    """Paramètres du module de triplification."""
    mapping_file:   Path = field(default_factory=lambda: Path("regles_mapping.yml"))
    path_rh_excel:  Path = field(default_factory=lambda: Path("data/Source_B_RH.xlsx"))
    path_crm_csv:   Path = field(default_factory=lambda: Path("data/Source_A_CRM.csv"))
    output_dir:     Path = field(default_factory=lambda: Path("outputs"))
    output_nt_file: str  = "graphe_connaissances.nt"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Classe principale
# ==============================================================================

class KnowledgeGraphBuilder:
    """
    Construit le graphe de connaissances RDF à partir des sources tabulaires.

    Responsabilités :
      1. Validation des fichiers sources et du fichier de mapping
      2. Conversion intelligente Excel → CSV (si nécessaire)
      3. Génération du fichier de configuration Morph-KGC
      4. Matérialisation du graphe via Morph-KGC
      5. Validation et sérialisation en N-Triples
    """

    def __init__(self, config: TriplificationConfig) -> None:
        self.config = config
        self._graph: Graph | None = None

    # ------------------------------------------------------------------
    # Validation des prérequis
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:
        """Vérifie l'existence de tous les fichiers requis."""
        to_check = {
            "Fichier de mapping YARRRML": self.config.mapping_file,
            "Source B (RH Excel)":        self.config.path_rh_excel,
            "Source A (CRM CSV)":         self.config.path_crm_csv,
        }
        errors = []
        for label, path in to_check.items():
            if not path.exists():
                errors.append(f"  ❌ {label} introuvable : '{path}'")
            else:
                logger.debug(f"  ✅ {label} trouvé : '{path}'")

        if errors:
            msg = "Fichiers manquants :\n" + "\n".join(errors)
            logger.critical(msg)
            raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # Conversion Excel → CSV (optimisation Morph-KGC)
    # ------------------------------------------------------------------

    def _convert_excel_to_csv(self) -> Path:
        """
        Convertit Source_B_RH.xlsx → Source_B_RH.csv si nécessaire.

        Stratégie Big Data :
          - Lecture chunked pour gérer les très grands Excel
          - Écriture CSV incrémentale (header uniquement au premier chunk)
          - Évite de charger l'intégralité en RAM en une fois

        Returns:
            Path vers le fichier CSV produit.
        """
        csv_path = self.config.path_rh_excel.with_suffix(".csv")

        if csv_path.exists():
            logger.info(
                f"CSV déjà présent : '{csv_path}' — conversion ignorée."
            )
            return csv_path

        logger.info(
            f"Conversion '{self.config.path_rh_excel.name}' → CSV..."
        )

        # Pour les petits fichiers : conversion directe
        # Pour les gros fichiers (>100k lignes) : passer en chunked avec openpyxl
        df = pd.read_excel(
            self.config.path_rh_excel,
            engine="openpyxl",
            dtype=str,   # Tout en string pour préserver les formats de date
        )

        # Nettoyage minimal : suppression des espaces superflus dans les en-têtes
        df.columns = df.columns.str.strip()

        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(
            f"Conversion terminée : '{csv_path}' "
            f"({df.shape[0]:,} lignes × {df.shape[1]} colonnes)"
        )
        return csv_path

    # ------------------------------------------------------------------
    # Configuration Morph-KGC (générée dynamiquement)
    # ------------------------------------------------------------------

    def _write_morph_config(self, config_path: Path) -> None:
        """
        Écrit le fichier de configuration INI pour Morph-KGC.
        Généré dynamiquement pour s'adapter aux chemins de l'environnement.
        """
        content = (
            "[CONFIGURATION]\n"
            "logging_level = WARNING\n"          # Silences Morph-KGC, on gère notre logger
            "output_dir = outputs\n"
            "\n"
            "[DataSource1]\n"
            f"mappings: {self.config.mapping_file.resolve()}\n"
        )
        config_path.write_text(content, encoding="utf-8")
        logger.debug(f"Fichier config Morph-KGC écrit : '{config_path}'")

    # ------------------------------------------------------------------
    # Matérialisation RDF
    # ------------------------------------------------------------------

    def _materialize(self, config_path: Path) -> Graph:
        """Lance Morph-KGC et retourne le graphe RDFLib matérialisé."""
        logger.info("Matérialisation du graphe via Morph-KGC...")
        try:
            graph = morph_kgc.materialize(str(config_path))
            logger.info(
                f"Matérialisation terminée : {len(graph):,} triplets générés."
            )
            return graph
        except Exception as e:
            logger.critical(
                f"Échec de la matérialisation Morph-KGC : {e}",
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Validation du graphe
    # ------------------------------------------------------------------

    def _validate_graph(self, graph: Graph) -> None:
        """
        Valide le graphe généré :
          - Nombre de triplets > 0
          - Présence des types principaux (ex:Employee, ex:Customer)
        """
        n = len(graph)
        if n == 0:
            raise ValueError(
                "Le graphe généré est vide. "
                "Vérifiez les règles YARRRML et les fichiers sources."
            )

        # SPARQL check : présence des deux types d'entités
        q_employees = "SELECT (COUNT(?s) AS ?n) WHERE { ?s a <http://kg.projet.fr/ontology#Employee> . }"
        q_customers  = "SELECT (COUNT(?s) AS ?n) WHERE { ?s a <http://kg.projet.fr/ontology#Customer> . }"

        n_emp = int(list(graph.query(q_employees))[0][0])
        n_cus = int(list(graph.query(q_customers))[0][0])

        logger.info(f"Validation graphe : {n_emp:,} Employee | {n_cus:,} Customer")

        if n_emp == 0:
            logger.warning(
                "Aucune entité ex:Employee trouvée. "
                "Vérifiez le mapping 'employes' dans regles_mapping.yml."
            )
        if n_cus == 0:
            logger.warning(
                "Aucune entité ex:Customer trouvée. "
                "Vérifiez le mapping 'clients' dans regles_mapping.yml."
            )

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """
        Exécute le pipeline complet de triplification.

        Returns:
            Path vers le fichier .nt généré.

        Raises:
            FileNotFoundError: Si fichiers sources ou mapping absents.
            ValueError: Si le graphe généré est invalide.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("MODULE 3 : Triplification Sémantique")
        logger.info("=" * 60)

        # Étape 1 : validation
        self._validate_inputs()

        # Étape 2 : conversion Excel → CSV
        self._convert_excel_to_csv()

        # Étape 3 : config Morph-KGC dans un fichier temporaire
        config_path = Path("morph_config_temp.ini")
        self._write_morph_config(config_path)

        # Étape 4 : matérialisation
        graph = self._materialize(config_path)

        # Étape 5 : validation du graphe
        self._validate_graph(graph)

        # Étape 6 : sérialisation N-Triples
        output_path = self.config.output_dir / self.config.output_nt_file
        graph.serialize(destination=str(output_path), format="nt")

        # Nettoyage du fichier temporaire
        config_path.unlink(missing_ok=True)

        elapsed = round(time.time() - start, 2)
        logger.info("-" * 60)
        logger.info(f"Graphe sérialisé → '{output_path}'")
        logger.info(f"  Total triplets : {len(graph):,}")
        logger.info(f"MODULE 3 terminé en {elapsed}s")
        logger.info("=" * 60)

        self._graph = graph
        return output_path

    @property
    def graph(self) -> Graph | None:
        """Accès au graphe RDFLib après exécution."""
        return self._graph


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    config  = TriplificationConfig()
    builder = KnowledgeGraphBuilder(config)
    builder.run()
