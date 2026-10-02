# R?sultat pilote D/C ? 2 octobre 2026

Protocole fix? dans `src/synthetic-audit-protocol.md` avant lancement. 10 seeds, 4 corr?lations, 2 repr?sentations, 2 readouts fixes, 3 concepts, 2 doses et 4 m?thodes : 3 840 mesures appari?es. Ce ne sont pas 3 840 r?p?titions ind?pendantes.

## Concept C2 exactement inutilis? par les mod?les

| Corr?lation | M?thode | Taux de faux positifs | Effet absolu moyen |
|---|---|---:|---:|
| 0 | constrained_probe | 0% | 1.17947e-16 |
| 0 | logistic_cav | 0% | 0.00729478 |
| 0 | oracle | 0% | 5.07901e-17 |
| 0 | prototype | 30% | 0.0163298 |
| 0.9 | constrained_probe | 0% | 3.44512e-16 |
| 0.9 | logistic_cav | 100% | 0.336763 |
| 0.9 | oracle | 0% | 5.07901e-17 |
| 0.9 | prototype | 100% | 0.454 |
| 0.99 | constrained_probe | 0% | 2.40279e-16 |
| 0.99 | logistic_cav | 100% | 0.486932 |
| 0.99 | oracle | 0% | 5.07901e-17 |
| 0.99 | prototype | 100% | 0.498495 |

## D?cision pr?fix?e

**No-go du pilote** : le contr?le sans corr?lation du prototype d?passe le taux maximal de faux positifs de 20 %. Le seuil relatif de 5 % est sensible aux fluctuations d??chantillonnage dans ce contr?le. Cette explication est une hypoth?se ? tester s?par?ment, pas une raison de changer le seuil apr?s coup.

La correction satisfait les crit?res de faux positifs et d?erreur dans ce cas lin?aire inversible. Ce r?sultat est attendu puisque les concepts observ?s suffisent ? ajuster un d?codeur lin?aire exact. Il ne d?montre pas une correction g?n?rale du confounding.

Le CAV logistique est test? s?par?ment : ses r?sultats varient avec la corr?lation et la repr?sentation. Ne pas g?n?raliser ? tous les CAV ou ? TCAV ; ce pilote mesure des d?placements finis calibr?s, pas le score de signe TCAV.

Le pooling ignore la connectivit? : le pilote n??tablit aucun avantage sp?cifique aux graphes ni robustesse sur architectures GNN entra?n?es. Il n?autorise donc pas un Go pour un papier de conf?rence principale.

## Travail requis avant engagement

1. R?pliquer avec deux v?ritables architectures GNN et une v?rit? fonctionnelle v?rifi?e (oracle masqu? et contr?le explicitement inutilis?), puis ajouter des GNN entra?n?s libres dont la d?pendance effective est mesur?e par l?oracle.
2. Ajouter bruit des annotations, concepts non lin?aires, variations de topologie et contr?le de d?rive non cible.
3. Tester calibration du diagnostic nul et intervalles sur seeds dans un nouveau protocole ind?pendant ; conserver le pr?sent ?chec.
4. Comparer aux m?thodes causales/conceptuelles ant?rieures et ?tablir ce que la topologie des graphes apporte.

Sources et param?tres complets : `src/synthetic-audit-protocol.md`. Aucun artefact BRACS ni GINE existant n?a ?t? modifi?.
