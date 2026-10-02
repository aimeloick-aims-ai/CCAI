---
noteId: "e9425c00bdd511f1a2c2df2d8bc5c738"
tags: []

---

# Protocole GCIA corrigé — v2

Protocole fixé avant modification des scripts, le 1 octobre 2026.

- Conserver sans modification `results/gine_baseline.pt`, le scaler et les splits.
- Concepts nodaux : area (23), circularity (27), mean_intensity_gray_scale (90), mean_contrast (55). Les unités restent celles des données ; ne pas supposer des unités physiques pour les features nodales sans métadonnées explicites.
- Concept relationnel : `spatial_distance`, moyenne des distances des arêtes en µm. Une dose positive augmente la distance. La connectivité est fixe : intervention sur un attribut, pas reconstruction géométrique.
- Les cibles nodales des probes sont les valeurs individuelles des cellules ; les résumés antérieurs étaient des moyennes par graphe. Leurs quantiles ne sont donc pas directement comparables.
- Les CAV et leurs scalers sont ajustés uniquement sur TRAIN. q20/q80 sont calculés uniquement sur les cellules TRAIN sélectionnées (50 maximum par graphe), ou les moyennes TRAIN pour la distance. Ces seuils sont enregistrés et réutilisés pour évaluer les probes sur VALIDATION et TEST, sans réajustement ni équilibrage des groupes d'évaluation.
- Vérifier la disjonction des identifiants des trois splits avant tout audit.
- Doses fixes : -2, -1, -0.5, 0, 0.5, 1, 2 écarts-types TRAIN. À zéro, retour d'une copie identique avant tout clipping. Tester features, attributs et indices d'arêtes, logits et prédictions.
- Clipping des doses non nulles : borne standardisée +/-5 élargie pour contenir la valeur initiale, avec plancher brut zéro pour area, circularity, grayscale, contraste et distance ; grayscale plafonnée à 255 en conservant toute valeur initiale déjà hors borne. Cela évite que le clipping déplace une valeur initiale en sens inverse de la dose. Ne pas plafonner circularity à 1 : le profilage contient déjà des valeurs supérieures.
- Enregistrer changement demandé, changement réalisé et taux de clipping, par dose et par contrôle. Les variations sont exprimées en unités standardisées. Les autres colonnes et la connectivité restent identiques. Les perturbations indépendantes de descripteurs peuvent être hors variété biologique ; ne pas les présenter comme interventions causales sur les images.
- Dix CAV issus de permutations des labels TRAIN. Sauvegarder distribution TCAV, moyenne, écart-type et quantiles 2.5/97.5 %. Ces quantiles décrivent les contrôles, pas un intervalle de confiance.
- Dix répétitions de contrôles d'intervention à signes aléatoires, avec signes fixes entre doses pour une répétition. Sauvegarder chaque répétition, moyenne, écart-type, quantiles et effet conceptuel moins moyenne aléatoire. Vérifier la magnitude réalisée après clipping ; ne pas supposer l'appariement exact après saturation.
- Évaluer et enregistrer séparément les performances CAV TRAIN et VALIDATION/TEST. Évaluation par graphes disjoints, seuils TRAIN fixes, aucune sélection de paramètres selon TEST.
- Garder les anciens `.npz` intacts ; sorties v2 distinctes et refus d'écraser une sortie existante.
- Garde-fous obligatoires avant exécution complète : mapping conforme au profilage, valeurs brutes finies, identité à zéro et direction monotone des doses.
- Exécuter Phase 1, inspecter sa validation, puis Phase 2 et inspecter sa validation. Les statistiques TEST sont descriptives finales ; aucune adaptation du protocole à leurs résultats.
