# Approches ATR 72 pour le jumeau 3D (`simulateur_3d_depannage`)

Ces fichiers sont des approches **simulées** (6 degrés de liberté, autoland classique, preset
`atr72`) écrites dans les conventions de `interface/vol.js` du jumeau, qui joue aujourd'hui une
approche écrite à la main (`Vol.lancer('approche')`, V_APP 107 kt, pente 3°, point de visée 300 m).

```bash
python -m src.viz.export_jumeau_trajectory --crosswind-kt 15 --gust-pct 50 --seed 2029 \
    --out docs/jumeau/approche_atr72_xw15_g050.json
```

## Format

```json
{
  "schema": "atr72-approche-jumeau",
  "aircraft": {"preset": "atr72", "name": "ATR 72-600", "vapp_mps": 55.0, "wheel_height_m": 2.55},
  "piste": {"longueur_m": 2500, "largeur_m": 45, "visee_m": 300, "pente_deg": 3.0},
  "conditions": {"vent_travers_kt": 15, "vent_de_deg": 90, "rafales_pct": 50, "graine": 2029},
  "dt": 0.05,
  "resultat": {"pose": true, "vz_toucher_mps": -0.88, "x_toucher_m": 301, "crabe_toucher_deg": 2.2},
  "frames": [{"t": 0.0, "x": -3500.0, "y": 0.0, "h": 198.5, "psi": -0.036, "theta": 0.058, "phi": -0.025,
              "V": 56.7, "Vsol": 61.7, "vz": -2.97, "gamma": -0.048, "sol": false,
              "cmd": {"mancheT": 0.0, "mancheR": 0.0, "palonnier": 0.0, "pl": 0.33, "volets": 1, "train": 1},
              "vent": {"total_mps": [3.4, -12.2, 0.0]}}]
}
```

- Repère du jumeau : `x` le long de la piste depuis le seuil (m), `y` à droite (m), `h` hauteur des
  **roues** (m, `h_cg` pour le centre de gravité), angles en radians (`psi` cap relatif à la piste,
  `theta` assiette, `phi` inclinaison), `V` vitesse air et `Vsol` vitesse sol (m/s), `vz` vitesse
  verticale (m/s, positive vers le haut), `sol` vrai dès que les roues touchent.
- Commandes du poste comme le `cmd` de `vol.js` : `mancheT` (tirer positif) et `mancheR` (droite
  positive) dans [-1, 1], `palonnier` (droite positive) dans [-1, 1], manette de puissance `pl` dans
  [0, 1] (0 = ralenti vol, 1 = pleine puissance ; le jumeau la ramène sur ses crans), `volets` et
  `train` sortis.
- `vent.total_mps` : vent total (moyen + turbulence + rafale) en NED au niveau de l'avion.

## Rejeu dans le jumeau : `vol_rejeu.patch`

`vol_rejeu.patch` modifie `interface/vol.js` du jumeau (il s'applique proprement sur sa branche
`main` du 23 octobre 2026 : `git apply vol_rejeu.patch` à la racine de `simulateur_3d_depannage`).
Il ajoute à `Vol` :

- `chargerApproche(doc)` : charge un document de ce schéma (renvoie `false` et garde les lois écrites
  à la main si le schéma n'est pas reconnu) ;
- `lancer('approche')` reprend alors l'enregistrement à 1000 m du point de visée (là où la finale
  écrite à la main commençait), gelé comme avant (`APP INIT REPOS`) ;
- `lancer('atterrissage')` rejoue les trames (interpolation linéaire, position directe, à-coup du
  toucher calculé sur la vitesse verticale enregistrée, manette ramenée entre les crans FI et montée) ;
  quand les trames sont épuisées (3 s après le toucher), le freinage jusqu'à l'arrêt reste celui du jumeau ;
- `rejoue` (lecture seule) dit si la phase courante vient de l'enregistrement.

La base de mouvement (`_indices`), le visuel (`dessiner`) et la lecture `vitesseKt` / `hauteurFt`
ne changent pas : ils lisent `a` et `cmd` comme avant. Les autres phases (piste, décollage, virage,
palier) ne sont pas touchées. Sans `chargerApproche`, le comportement est identique à l'original.

Pour s'en servir dans `lancement.js`, après `this.vol = new Vol()` :

```js
fetch('approches/approche_atr72_xw15_g050.json').then(r => r.json()).then(doc => this.vol?.chargerApproche(doc));
```

Test hors navigateur (Node, module copié en `.mjs`) sur `approche_atr72_xw15_g050.json` : finale
reprise à x = −700 m, 112 kt, 169 ft ; toucher à t = 17,4 s à x = 301 m, 2,2° de crabe résiduel,
à-coup vertical 0,62 ; trames épuisées à x = 459 m, arrêt par le freinage du jumeau à x = 917 m.
