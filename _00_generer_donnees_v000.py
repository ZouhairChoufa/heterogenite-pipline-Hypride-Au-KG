"""
_00_generer_donnees_v000.py
===========================
Module 0 (Version Hybride) : Génération à partir de données réelles.

Ce script ingère le dataset réel 'online_retail_II.csv', extrait les vrais
clients et leur vrai Chiffre d'Affaires (Quantité * Prix), puis utilise Faker
pour simuler les attributs manquants (Noms, Dates de naissance, Info RH) 
afin de créer l'hétérogénéité requise pour le pipeline.
"""

import time
import random
from pathlib import Path
import pandas as pd
from faker import Faker
from utils.logger import get_logger

logger = get_logger(__name__)

class HybridDataGenerator:
    def __init__(self, input_file: str = "data/online_retail_II.csv", output_dir: str = "data"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.faker = Faker('fr_FR')
        # Fixer la graine pour toujours générer les mêmes noms pour les mêmes ID
        Faker.seed(42) 
        random.seed(42)

    def run(self):
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("MODULE 0 (V000) : Génération Hybride (Réel + Synthétique)")
        logger.info("=" * 60)

        if not self.input_file.exists():
            logger.critical(f"Fichier introuvable : {self.input_file}")
            return

        # 1. Extraction des données réelles
        logger.info("1. Lecture du fichier transactionnel réel...")
        df_real = pd.read_csv(self.input_file)
        
        # Filtrer les lignes sans ID client et calculer le revenu par ligne
        df_real = df_real.dropna(subset=['Customer ID'])
        df_real['Customer ID'] = df_real['Customer ID'].astype(int)
        df_real['Revenue'] = df_real['Quantity'] * df_real['Price']

        # Agréger par client pour obtenir le vrai CA total
        logger.info("2. Calcul du Chiffre d'Affaires réel par client...")
        df_customers = df_real.groupby('Customer ID').agg({
            'Revenue': 'sum',
            'Country': 'first'
        }).reset_index()

        # Filtrer les revenus négatifs (retours de commandes)
        df_customers = df_customers[df_customers['Revenue'] > 0]
        n_clients = len(df_customers)
        logger.info(f"   -> {n_clients:,} clients uniques extraits avec succès.")

        # 3. Augmentation des données avec Faker (Création des profils)
        logger.info("3. Enrichissement des profils (Noms, Dates, RH) avec Faker...")
        profiles = []
        for _, row in df_customers.iterrows():
            cid = row['Customer ID']
            revenue = round(row['Revenue'], 2)
            
            profiles.append({
                'real_id': cid,
                'revenue': revenue,
                'country': row['Country'],
                'first_name': self.faker.first_name(),
                'last_name': self.faker.last_name(),
                'dob': self.faker.date_of_birth(minimum_age=22, maximum_age=65),
                'email': self.faker.email(),
                'city': self.faker.city(),
                'dept': random.choice(['Ventes', 'Marketing', 'IT', 'Logistique', 'Finance']),
                'contract': random.choice(['CDI', 'CDD', 'Freelance']),
                'hire_date': self.faker.date_between(start_date='-10y', end_date='today'),
                'salary': round(random.uniform(2500, 8500), 2)
            })

        # 4. Création des deux sources hétérogènes
        logger.info("4. Séparation en sources hétérogènes (CRM et RH)...")
        
        # -- SOURCE A : CRM (Format CSV, Dates FR, Noms complets) --
        source_a = pd.DataFrame({
            'ID_Client': [f"C-{p['real_id']}" for p in profiles],
            'dt_naissance': [p['dob'].strftime('%d/%m/%Y') for p in profiles],
            'nom_complet': [f"{p['first_name']} {p['last_name']}" for p in profiles],
            'CA_annuel': [str(p['revenue']).replace('.', ',') for p in profiles],
            'email': [p['email'] for p in profiles],
            'ville': [p['city'] for p in profiles]
        })

        # -- SOURCE B : RH (Format Excel, Dates ISO, Noms séparés) --
        # On mélange l'ordre pour rendre le matching plus réaliste
        random.shuffle(profiles) 
        source_b = pd.DataFrame({
            'Matricule_RH': [f"RH{p['real_id']}" for p in profiles],
            'date_nais': [p['dob'].strftime('%Y-%m-%d') for p in profiles],
            'last_name': [p['last_name'] for p in profiles],
            'first_name': [p['first_name'] for p in profiles],
            'chiffre_affaire': [p['revenue'] for p in profiles],
            'departement': [p['dept'] for p in profiles],
            'contrat': [p['contract'] for p in profiles],
            'salaire_mensuel': [p['salary'] for p in profiles],
            'date_embauche': [p['hire_date'].strftime('%Y-%m-%d') for p in profiles]
        })

        # 5. Sauvegarde
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path_a = self.output_dir / "Source_A_CRM.csv"
        path_b = self.output_dir / "Source_B_RH.xlsx"

        source_a.to_csv(path_a, index=False, sep=';', encoding='utf-8')
        source_b.to_excel(path_b, index=False)

        elapsed = round(time.time() - start_time, 2)
        logger.info("-" * 60)
        logger.info(f"✅ Fichier CRM généré : {path_a} ({len(source_a):,} lignes)")
        logger.info(f"✅ Fichier RH généré  : {path_b} ({len(source_b):,} lignes)")
        logger.info(f"MODULE 0 terminé en {elapsed}s")
        logger.info("=" * 60)

if __name__ == "__main__":
    generator = HybridDataGenerator()
    generator.run()