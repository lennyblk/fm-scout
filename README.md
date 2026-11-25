## Installation sur Mac (Apple Silicon)

1. **Copier le projet.** Copie le dossier `fm-scout` (clé USB, GitHub privé...) **sans le dossier
   `.venv`**, propre à Windows. Mets-le par exemple dans `~/Documents/fm-scout`.

2. **Python 3.12 ou plus.** Dans le Terminal :

   ```bash
   python3 --version
   ```

   S'il est absent ou plus ancien, installe-le depuis <https://www.python.org/downloads/>, puis
   rouvre le Terminal.

3. **Installer fm-scout :**

   ```bash
   cd ~/Documents/fm-scout
   python3 -m venv .venv
   .venv/bin/python -m pip install -e ".[dev,fmsave]"
   .venv/bin/python -m pytest -q
   ```

   La dernière commande doit finir par « passed ».

4. **Installer l'IA :** <https://ollama.com/download>, ouvre l'application une fois, puis :

   ```bash
   ollama pull qwen3:8b
   ollama create fm-scout -f Modelfile.mac
   .venv/bin/python eval_tools.py
   ```

   Environ 5 Go à télécharger. Vise au moins 10/12 sur « Outil + arguments ».

5. **Vérifier ta partie** (sauvegarde d'abord dans FM27) :

   ```bash
   .venv/bin/fm-scout inspect
   ```

   Tu dois voir ton club, la date et ton budget.

## Installation sur Windows

Python 3.12+ et [Ollama](https://ollama.com/download) installés, dans PowerShell :

```powershell
cd C:\chemin\vers\fm-scout
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,fmsave]"
.venv\Scripts\python -m pytest -q
ollama pull qwen3.5:4b
ollama create fm-scout -f Modelfile
.venv\Scripts\python eval_tools.py
.venv\Scripts\fm-scout inspect
```

## Utilisation

**Sauvegarde dans FM27 avant**

| Je veux                          | Commande                                                                            |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| Discuter avec le scout (IA)      | `fm-scout --save ajax`                                                              |
| Poser une seule question (IA)    | `fm-scout --save ajax ask "qui recruter cet été ?"`                                 |
| Postes à renforcer et recrues    | `fm-scout --save ajax recommend`                                                    |
| ... autre budget ou autre saison | `fm-scout --save ajax recommend --budget 40M --season 2`                            |
| Chercher des joueurs             | `fm-scout --save ajax search --pos MDC --max-age 21 --max-fee 5M --role sentinelle` |
| Comparer                         | `fm-scout --save ajax compare "Nom 1" "Nom 2" --role sentinelle`                    |
| Voir mon effectif                | `fm-scout --save ajax needs`                                                        |
| Voir les rôles                   | `fm-scout roles`                                                                    |

- Pendant une discussion, une nouvelle sauvegarde est rechargée avant ta question suivante (la
  conversation repart alors de zéro). Les autres commandes relisent la save à chaque lancement.
- Seuls `ask` et la discussion utilisent l'IA.
- `--save ajax` prend la save la plus récente dont le nom contient « ajax ». Sans `--save`, la plus
  récente de toutes. Un chemin complet marche aussi.
- Saisons de `recommend` : 0 en cours, 1 prochaine (défaut), 2 suivante.
- `search --max-wage` : par semaine, ou `80K/mois`.

## Réglages

- `config.toml` : save par défaut, modèle, `money_rate`, `show_hidden`.
  - `money_rate` = montant affiché en jeu / montant lu. 1.167 a été mesuré sur une seule partie :
    vérifie-le avec le salaire d'un de tes joueurs si tu changes de partie.
  - `show_hidden = true` lit la vraie CA/PA, cachée par le jeu. `false` : le potentiel ne compte plus.
- `src/fm_scout/data/recommend.yaml` : poids, âge maximum, marge de prix, importance du potentiel.
- `src/fm_scout/data/roles.yaml` : attributs importants par rôle.
- `src/fm_scout/data/zones.yaml` : profondeur souhaitée pour `needs`.
- Variables d'environnement : `FM_SCOUT_SAVE`, `FM_SCOUT_MODEL`, `FM_SCOUT_LLM_URL`, `FM_SCOUT_CSV`.
