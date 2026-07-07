[200~python -c "
import json, os, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import pandas as pd

ONTOLOGIE = {
    'ex:employeeId': 'Identifiant unique de l employé ou du client',
    'schema:birthDate': 'Date de naissance de la personne',
    'foaf:familyName': 'Nom de famille de la personne',
    'foaf:firstName': 'Prénom de la personne',
    'ex:fullName': 'Nom complet (prénom + nom de famille concaténés)',
    'schema:email': 'Adresse email de contact',
    'ex:department': 'Département ou service dans l organisation',
    'ex:annualRevenue': 'Chiffre d affaires ou revenu annuel',
    'schema:jobTitle': 'Poste ou type de contrat professionnel',
    'schema:addressLocality': 'Ville ou localité',
}

GOLD = {
    'Matricule_RH': 'ex:employeeId',
    'Nom_Famille': 'foaf:familyName',
    'Prenom': 'foaf:firstName',
    'date_nais': 'schema:birthDate',
    'departement': 'ex:department',
    'contrat': 'schema:jobTitle',
    'salaire_mensuel': None,
    'date_embauche': None,
}

llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.0, api_key=os.environ['GROQ_API_KEY'])
parser = JsonOutputParser()
onto_texte = '\n'.join(f'  - {u} : {d}' for u,d in ONTOLOGIE.items())
known = set(ONTOLOGIE.keys()) | {'UNMATCHED'}

prompt = PromptTemplate(
    template=(
        'Tu es un expert en ontologies.\n'
        'Associe la colonne \"{colonne}\" (exemples: {exemples}) '
        'a la propriete parmi:\n{ontologie}\n\n'
        'Si aucune ne correspond retourne UNMATCHED.\n'
        'Reponds STRICTEMENT avec ce JSON et rien dautre, sans markdown:\n'
        '{{\"cible\": \"URI_ici\", \"confiance\": 0.9, \"justification\": \"raison courte\"}}\n'
        '{format_instructions}'
    ),
    input_variables=['colonne','exemples','ontologie'],
    partial_variables={'format_instructions': parser.get_format_instructions()},
)
chain = prompt | llm | parser

df = pd.read_excel('data/Source_B_RH.xlsx', engine='openpyxl')
results = []
llm_calls = 0
call_times = []

for col in df.columns.tolist():
    non_null = df[col].dropna()
    n = min(3, len(non_null))
    exemples = non_null.sample(n, random_state=42).tolist() if n > 0 else []
    t0 = time.time()
    try:
        res = chain.invoke({'colonne': col, 'exemples': json.dumps(exemples, ensure_ascii=False), 'ontologie': onto_texte})
        latency = round((time.time()-t0)*1000,1)
        call_times.append(latency)
        llm_calls += 1
        # FIX : gere list ET dict
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        if not isinstance(res, dict):
            raise ValueError(f'Format inattendu: {type(res)}')
        cible = str(res.get('cible','UNMATCHED')).strip()
        confiance = float(res.get('confiance', 0.0))
        justif = str(res.get('justification','')).strip()
        if cible not in known:
            cible = 'UNMATCHED'
        results.append({'source_column': col, 'target_uri': cible, 'confidence': round(confiance,4), 'method': 'LLM_ONLY', 'latency_ms': latency, 'justification': justif})
        icon = '✅' if cible != 'UNMATCHED' else '❌'
        print(f'  {icon} LLM | {col} → {cible} (conf={confiance:.3f}, {latency}ms)')
    except Exception as e:
        results.append({'source_column': col, 'target_uri': 'UNMATCHED', 'confidence': 0.0, 'method': 'LLM_ONLY', 'justification': str(e)})
        print(f'  ❌ ERREUR {col} : {e}')

# Calcul métriques
tp=fp=fn=tn=0
for r in results:
    col=r['source_column']; pred=r['target_uri']; exp=GOLD.get(col)
    if exp is None:
        if pred=='UNMATCHED': tn+=1
        else: fp+=1
    else:
        if pred=='UNMATCHED': fn+=1
        elif pred==exp: tp+=1
        else: fp+=1

precision = tp/(tp+fp) if (tp+fp)>0 else 0
recall = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
avg_lat = round(sum(call_times)/len(call_times),1) if call_times else 0

print()
print('══════════════════════════════════════')
print('  VARIANTE B (CORRIGÉE) — RÉSULTATS')
print('══════════════════════════════════════')
print(f'  Précision  : {precision:.4f}')
print(f'  Rappel     : {recall:.4f}')
print(f'  F1-score   : {f1:.4f}')
print(f'  Appels LLM : {llm_calls}')
print(f'  Latence moy: {avg_lat}ms/appel')
print(f'  TP={tp} FP={fp} FN={fn} TN={tn}')
print('══════════════════════════════════════')

Path('outputs/ablation').mkdir(parents=True, exist_ok=True)
Path('outputs/ablation/ablation_llm_only_FIXED.json').write_text(
    json.dumps({'variant':'B_LLM_Only_FIXED','results':results,'metrics':{'precision':round(precision,4),'recall':round(recall,4),'f1':round(f1,4),'TP':tp,'FP':fp,'FN':fn,'TN':tn},'llm_calls':llm_calls,'avg_latency_ms':avg_lat}, indent=2, ensure_ascii=False), encoding='utf-8')
print('  Résultats sauvegardés → outputs/ablation/ablation_llm_only_FIXED.json')
clear 
