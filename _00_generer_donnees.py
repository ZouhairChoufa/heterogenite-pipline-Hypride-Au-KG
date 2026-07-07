"""
00_generer_donnees.py
=====================
Module 0 : Génération de données factices hétérogènes.

Produit deux sources volontairement différentes (hétérogénéité de schéma) :
  - data/Source_A_CRM.csv  : données clients (CSV, séparateur ';')
  - data/Source_B_RH.xlsx  : données employés (Excel)

Chevauchement contrôlé : OVERLAP_RATIO des personnes apparaissent dans les deux
sources (avec des noms de colonnes différents → challenge du schema matching).
"""

import time
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from faker import Faker

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class DataGenConfig:
    """Paramètres de génération des jeux de données synthétiques."""
    n_crm: int        = 4_000     # Lignes dans Source_A_CRM
    n_rh: int         = 3_000     # Lignes dans Source_B_RH
    overlap_ratio: float = 0.40   # 40 % des personnes sont dans les deux sources
    locale: str       = "fr_FR"
    seed: int         = 42
    output_dir: Path  = field(default_factory=lambda: Path("data"))

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Générateur
# ==============================================================================

class SyntheticDataGenerator:
    """
    Génère deux DataFrames avec une hétérogénéité de schéma intentionnelle
    et un chevauchement partiel contrôlé.

    Hétérogénéités introduites :
      - Noms de colonnes différents (nom_complet vs first_name + last_name)
      - Formats de date différents (%d/%m/%Y vs ISO 8601)
      - Identifiants différents (ID_Client vs Matricule_RH)
      - Colonnes exclusives à chaque source (email, departement…)
    """

    def __init__(self, config: DataGenConfig) -> None:
        self.config = config
        self.fake = Faker(config.locale)
        Faker.seed(config.seed)
        random.seed(config.seed)
        logger.info(f"Générateur initialisé (seed={config.seed}, locale={config.locale})")

    # ------------------------------------------------------------------
    # Génération d'un pool de personnes partagées
    # ------------------------------------------------------------------

    def _generate_person_pool(self, n: int) -> list[dict]:
        """Crée un pool de personnes réutilisable entre les deux sources."""
        return [
            {
                "first_name": self.fake.first_name(),
                "last_name":  self.fake.last_name(),
                "birthdate":  self.fake.date_of_birth(minimum_age=22, maximum_age=65),
            }
            for _ in range(n)
        ]

    # ------------------------------------------------------------------
    # Source A — CRM (CSV)
    # ------------------------------------------------------------------

    def generate_crm(self, shared_persons: list[dict]) -> pd.DataFrame:
        """
        Génère Source_A_CRM avec les colonnes :
        ID_Client, nom_complet, email, dt_naissance, ville, ca_annuel, segment
        """
        logger.info(f"Génération CRM : {self.config.n_crm} lignes...")
        records = []

        # Personnes partagées avec le RH (chevauchement)
        n_shared = int(self.config.n_crm * self.config.overlap_ratio)
        shared_sample = random.sample(shared_persons, min(n_shared, len(shared_persons)))

        for i, person in enumerate(shared_sample):
            records.append({
                "ID_Client":    f"CRM-{i+1:05d}",
                "nom_complet":  f"{person['first_name']} {person['last_name']}",
                "email":        self.fake.email(),
                "dt_naissance": person["birthdate"].strftime("%d/%m/%Y"),  # Format FR
                "ville":        self.fake.city(),
                "ca_annuel":    round(random.uniform(10_000, 500_000), 2),
                "segment":      random.choice(["Gold", "Silver", "Bronze", "Platinum"]),
            })

        # Personnes exclusives CRM
        for i in range(self.config.n_crm - len(shared_sample)):
            records.append({
                "ID_Client":    f"CRM-{len(shared_sample)+i+1:05d}",
                "nom_complet":  f"{self.fake.first_name()} {self.fake.last_name()}",
                "email":        self.fake.email(),
                "dt_naissance": self.fake.date_of_birth(
                    minimum_age=22, maximum_age=65
                ).strftime("%d/%m/%Y"),
                "ville":        self.fake.city(),
                "ca_annuel":    round(random.uniform(10_000, 500_000), 2),
                "segment":      random.choice(["Gold", "Silver", "Bronze", "Platinum"]),
            })

        df = pd.DataFrame(records)
        logger.debug(f"CRM généré : {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Source B — RH (Excel)
    # ------------------------------------------------------------------

    DEPARTMENTS = [
        "Informatique", "Ressources Humaines", "Finance",
        "Marketing", "Juridique", "Logistique", "R&D",
    ]
    CONTRACTS = ["CDI", "CDD", "Stage", "Alternance", "Freelance"]

    def generate_rh(self, shared_persons: list[dict]) -> pd.DataFrame:
        """
        Génère Source_B_RH avec les colonnes :
        Matricule_RH, Nom_Famille, Prenom, date_nais, departement, contrat,
        salaire_mensuel, date_embauche
        (Schéma intentionnellement différent du CRM)
        """
        logger.info(f"Génération RH : {self.config.n_rh} lignes...")
        records = []

        # Personnes partagées avec le CRM
        n_shared = int(self.config.n_rh * self.config.overlap_ratio)
        shared_sample = random.sample(shared_persons, min(n_shared, len(shared_persons)))

        for i, person in enumerate(shared_sample):
            records.append({
                "Matricule_RH":    f"RH-{i+1:05d}",
                "Nom_Famille":     person["last_name"],           # Séparé (vs nom_complet)
                "Prenom":          person["first_name"],
                "date_nais":       person["birthdate"].strftime("%Y-%m-%d"),  # Format ISO
                "departement":     random.choice(self.DEPARTMENTS),
                "contrat":         random.choice(self.CONTRACTS),
                "salaire_mensuel": random.randint(1_800, 8_500),
                "date_embauche":   self.fake.date_between(
                    start_date="-10y", end_date="today"
                ).strftime("%Y-%m-%d"),
            })

        # Personnes exclusives RH
        for i in range(self.config.n_rh - len(shared_sample)):
            records.append({
                "Matricule_RH":    f"RH-{len(shared_sample)+i+1:05d}",
                "Nom_Famille":     self.fake.last_name(),
                "Prenom":          self.fake.first_name(),
                "date_nais":       self.fake.date_of_birth(
                    minimum_age=22, maximum_age=65
                ).strftime("%Y-%m-%d"),
                "departement":     random.choice(self.DEPARTMENTS),
                "contrat":         random.choice(self.CONTRACTS),
                "salaire_mensuel": random.randint(1_800, 8_500),
                "date_embauche":   self.fake.date_between(
                    start_date="-10y", end_date="today"
                ).strftime("%Y-%m-%d"),
            })

        df = pd.DataFrame(records)
        logger.debug(f"RH généré : {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Pipeline complet
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Génère et persiste les deux sources de données.

        Returns:
            Tuple (df_crm, df_rh)
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("MODULE 0 : Génération des données synthétiques")
        logger.info("=" * 60)

        # Pool de personnes communes (plus grand que les deux sources pour le sampling)
        pool_size = max(self.config.n_crm, self.config.n_rh)
        shared_persons = self._generate_person_pool(pool_size)
        logger.info(f"Pool de {pool_size:,} personnes partagées créé.")

        df_crm = self.generate_crm(shared_persons)
        df_rh  = self.generate_rh(shared_persons)

        # Persistance
        path_crm = self.config.output_dir / "Source_A_CRM.csv"
        path_rh  = self.config.output_dir / "Source_B_RH.xlsx"

        df_crm.to_csv(path_crm, sep=";", index=False, encoding="utf-8")
        df_rh.to_excel(path_rh, index=False, engine="openpyxl")

        elapsed = round(time.time() - start, 2)
        logger.info("-" * 60)
        logger.info(f"Source A (CRM) → {path_crm}  [{df_crm.shape[0]:,} lignes]")
        logger.info(f"Source B (RH)  → {path_rh} [{df_rh.shape[0]:,} lignes]")
        logger.info(
            f"Chevauchement estimé : ~{int(self.config.n_crm * self.config.overlap_ratio):,} "
            f"personnes communes"
        )
        logger.info(f"MODULE 0 terminé en {elapsed}s")
        logger.info("=" * 60)

        return df_crm, df_rh


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    config = DataGenConfig(n_crm=4_000, n_rh=3_000, overlap_ratio=0.40)
    generator = SyntheticDataGenerator(config)
    generator.run()
