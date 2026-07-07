from rdflib import Graph
from pyvis.network import Network
import networkx as nx

print("Chargement du graphe (cela peut prendre quelques secondes)...")
g = Graph()
# On charge les deux fichiers
g.parse("outputs/graphe_connaissances.nt", format="nt")
g.parse("outputs/liens_identite.nt", format="nt")

# Création du réseau visuel
net = Network(height='800px', width='100%', directed=True, notebook=False)

# Pour éviter de faire crasher le navigateur, on limite à 300 triplets
# Idéalement, filtrez avec une requête SPARQL sur un matricule précis
limite = 300
for i, (sujet, predicat, objet) in enumerate(g):
    if i >= limite: break

    # Nettoyage des URIs pour l'affichage (enlève les liens web longs)
    s_label = str(sujet).split('#')[-1].split('/')[-1]
    p_label = str(predicat).split('#')[-1].split('/')[-1]
    o_label = str(objet).split('#')[-1].split('/')[-1]

    net.add_node(s_label, title=str(sujet), color="#97C2FC") # Nœud source
    net.add_node(o_label, title=str(objet), color="#FFD194") # Nœud cible
    net.add_edge(s_label, o_label, title=p_label, label=p_label)

# Générer le fichier HTML interactif
net.show("mon_graphe_interactif.html", notebook=False)
print("✅ Visualisation générée dans 'mon_graphe_interactif.html' !")