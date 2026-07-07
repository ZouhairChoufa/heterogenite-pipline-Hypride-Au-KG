from rdflib import Graph, URIRef
from pyvis.network import Network

print("Chargement du Graphe Big Data (Patientez quelques secondes)...")
g = Graph()
g.parse("outputs/graphe_connaissances.nt", format="nt")
g.parse("outputs/liens_identite.nt", format="nt")

print("Recherche d'une réconciliation parfaite (owl:sameAs)...")
owl_sameas = URIRef("http://www.w3.org/2002/07/owl#sameAs")

# 1. On trouve au hasard 1 lien qui relie un Employé à un Client
employe_cible = None
client_cible = None

for s, p, o in g.triples((None, owl_sameas, None)):
    employe_cible = s
    client_cible = o
    break  # On s'arrête dès qu'on a trouvé un "Match"

# 2. On isole uniquement le sous-graphe de cette personne
sous_graphe = Graph()

# Ajouter les attributs RH
for s, p, o in g.triples((employe_cible, None, None)):
    sous_graphe.add((s, p, o))
    
# Ajouter les attributs CRM
for s, p, o in g.triples((client_cible, None, None)):
    sous_graphe.add((s, p, o))

# Ajouter le fameux pont "sameAs" au milieu
sous_graphe.add((employe_cible, owl_sameas, client_cible))

print(f"Génération de la vue 3D pour l'entité : {employe_cible}")

# 3. Génération visuelle
net = Network(height='800px', width='100%', directed=True, notebook=False)
net.force_atlas_2based() # Un algorithme physique très fluide

for sujet, predicat, objet in sous_graphe:
    s_label = str(sujet).split('#')[-1]
    p_label = str(predicat).split('#')[-1]
    o_label = str(objet).split('#')[-1]

    # Couleurs distinctes pour bien montrer la fusion
    if "Employee" in str(sujet):
        color = "#ff9999" # Rouge clair pour RH
    elif "Customer" in str(sujet):
        color = "#99ccff" # Bleu clair pour CRM
    else:
        color = "#dddddd"

    net.add_node(s_label, title=str(sujet), color=color, size=20)
    net.add_node(o_label, title=str(objet), color="#f2f2f2", size=15)
    
    # Si c'est le pont sameAs, on le met en rouge et plus épais !
    if "sameAs" in p_label:
        net.add_edge(s_label, o_label, title="RÉCONCILIATION SPLINK", label="owl:sameAs", color="red", width=3)
    else:
        net.add_edge(s_label, o_label, title=p_label, label=p_label, color="#999999")

net.show("mon_graphe_interactif_PFE.html", notebook=False)
print("✅ La présentation est prête : 'mon_graphe_interactif_PFE.html'")