# Expériences D puis C : protocole pilote

Fixé le 2 octobre 2026, avant exécution. Aucun changement au GINE BRACS.

## Question et vérité terrain

Une intervention latente visant un concept corrélé peut déplacer un autre concept réellement utilisé. Comparer son diagnostic à l'intervention explicite sur le concept, en conservant les autres concepts fixes.

La corrélation seule ne garantit pas un faux positif : modifier exactement une colonne inutilisée d'un modèle qui l'ignore produit zéro effet. Ne pas confondre une variable inutilisée par le générateur de labels avec une variable inutilisée par un modèle entraîné.

Le pilote utilise trois concepts gaussiens. C1 et C2 partagent une cause commune ; C3 est indépendant. Les modèles fixés utilisent C1 et C3 et ignorent exactement C2. Deux fonctions de réponse : linéaire et tanh. Les graphes contiennent huit nœuds avec bruit centré par graphe ; leur pooling moyen restitue les concepts. La représentation est soit directe, soit mélangée par une transformation inversible fixe.

Cela constitue une vérification analytique sur graphes à pooling, pas une validation sur plusieurs architectures GNN entraînées.

## Paramètres et méthodes fixés

- Dix seeds indépendants, 1 200 graphes TRAIN et 400 TEST synthétiques par seed.
- Corrélations : 0, 0.5, 0.9, 0.99. Doses symétriques +/-0.5, sans clipping.
- Prototype marginal : différence des représentations moyennes q80/q20 du TRAIN.
- CAV logistique : même fonction `fit_cav` que Phase 2, mêmes quantiles TRAIN et coefficient remis dans les coordonnées originales.
- Correction : probe linéaire multivarié h -> concepts, ajusté uniquement sur TRAIN. Résoudre une direction de norme minimale qui change la cible d'une unité et préserve les autres sorties du probe. Cette contrainte suppose des concepts observés et un probe adéquat ; elle n'identifie pas un SCM inconnu.
- Oracle : modifier directement le concept cible en conservant les autres fixes, puis reconstruire la représentation. L'oracle n'est pas fourni à la correction.
- Toutes les directions estimées sont calibrées au même changement cible selon le probe TRAIN. Rapporter aussi le changement réel de tous les concepts calculé avec le générateur, réservé à l'évaluation.

## Mesures et décision fixées

Mesure principale : effet absolu moyen sur le logit pour C2, dont l'effet oracle vaut exactement zéro. Faux positif si l'effet dépasse 5 % de l'effet oracle de C1 à la même dose. Rapporter erreur absolue versus oracle pour tous les concepts et dérive maximale non cible. Aucun choix de méthode ou dose selon TEST.

Go pilote : à rho=0.9 et 0.99, faux positif de la méthode marginale dans au moins 80 % des seeds pour chaque représentation et réponse, contre au plus 5 % pour la correction ; erreur de la correction <=10 % de l'effet oracle C1. Le contrôle rho=0 en représentation directe doit avoir au plus 20 % de faux positifs marginaux. Rapporter le CAV logistique séparément : un échec du prototype ne prouve pas un échec du CAV logistique.

Go papier : critères précédents ET reproduction sur au moins deux véritables architectures GNN avec dépendances fonctionnelles vérifiées, contrôles de bruit de concepts et interventions réalisables. Le pilote seul ne permet jamais un Go papier. Un résultat négatif est conservé et conduit à réviser l'hypothèse, pas les seuils a posteriori.

## Précédents ciblés

- [CaCE, 2019](https://arxiv.org/abs/1907.07165) : effets causaux de concepts sur des classifieurs.
- [Debiasing Concept-Based Explanations with Causal Analysis, ICLR 2021](https://assets.amazon.science/2a/60/aff3520f4d52bb195f4a674ae413/debiasing-concept-based-explanations-with-causal-analysis.pdf) : biais de concepts associés à de l'information confondante.
- [Addressing Leakage in Concept Bottleneck Models, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html) : information supplémentaire dans les probabilités conceptuelles.
- [A Closer Look at the Intervention Procedure of Concept Bottleneck Models, 2023](https://arxiv.org/abs/2302.14260) : évaluation d'interventions avec plusieurs graphes causaux synthétiques.
- [Stochastic Concept Bottleneck Models, 2024](https://arxiv.org/abs/2406.19272) : dépendances entre concepts et interventions.
- [GraphXAI / Evaluating Explainability for Graph Neural Networks](https://arxiv.org/abs/2208.09339) : benchmarks graphes avec explications de référence et pièges d'évaluation.

Recherche ciblée, pas revue systématique ni preuve de nouveauté. Le pilote doit préciser l'avantage propre aux graphes avant toute revendication de contribution principale.
