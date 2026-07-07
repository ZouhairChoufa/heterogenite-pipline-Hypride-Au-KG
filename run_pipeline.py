"""
run_pipeline.py
===============
Point d'entrée unique — exécute les 5 modules dans l'ordre.

Usage :
    python run_pipeline.py              # Pipeline complet
    python run_pipeline.py --from 2     # Reprend depuis le Module 2
    python run_pipeline.py --only 4     # Exécute uniquement le Module 4
"""

import argparse
import sys
import time

from utils.logger import get_logger

logger = get_logger("pipeline.main")


def run_full_pipeline(from_module: int = 0, only_module: int | None = None) -> None:
    start_total = time.time()
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     PFE BIBDA — Knowledge Graph Pipeline — START        ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    modules = {
        0: ("Module 0 : Génération des données",      "00_generer_donnees",     "run"),
        1: ("Module 1 : Ingestion & Profilage",        "01_module_ingestion",    "run"),
        2: ("Module 2 : Schema Matching",              "02_module_matching",     "run_schema_matching"),
        3: ("Module 3 : Triplification",               "03_module_triplification","run"),
        4: ("Module 4 : Résolution d'Entités",         "04_module_splink",       "resolve"),
    }

    # Filtre les modules à exécuter
    if only_module is not None:
        to_run = {only_module: modules[only_module]}
    else:
        to_run = {k: v for k, v in modules.items() if k >= from_module}

    for mod_id, (label, module_name, _) in to_run.items():
        logger.info(f"\n▶ Démarrage {label}...")
        try:
            if mod_id == 0:
                from _00_generer_donnees import DataGenConfig, SyntheticDataGenerator
                gen = SyntheticDataGenerator(DataGenConfig())
                gen.run()

            elif mod_id == 1:
                from _01_module_ingestion import DataIngestionProfiler, IngestionConfig
                DataIngestionProfiler(IngestionConfig()).run()

            elif mod_id == 2:
                from _02_module_matching import run_schema_matching
                run_schema_matching("data/Source_B_RH.xlsx")

            elif mod_id == 3:
                from _03_module_triplification import KnowledgeGraphBuilder, TriplificationConfig
                KnowledgeGraphBuilder(TriplificationConfig()).run()

            elif mod_id == 4:
                from _04_module_splink import EntityResolutionConfig, EntityResolver
                EntityResolver(EntityResolutionConfig()).resolve()

            logger.info(f"✅ {label} — SUCCÈS")

        except Exception as e:
            logger.critical(f"❌ {label} — ÉCHEC : {e}", exc_info=True)
            logger.critical("Pipeline interrompu. Consultez logs/pipeline.log pour le traceback complet.")
            sys.exit(1)

    elapsed = round(time.time() - start_total, 2)
    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║     Pipeline terminé en {elapsed}s                         ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG Pipeline PFE BIBDA")
    parser.add_argument("--from", dest="from_module", type=int, default=0,
                        help="Démarre depuis le module N (0-4)")
    parser.add_argument("--only", dest="only_module", type=int, default=None,
                        help="Exécute uniquement le module N")
    args = parser.parse_args()
    run_full_pipeline(from_module=args.from_module, only_module=args.only_module)
