Výzkumný projekt pro automatizované trasování vedení VVN pomocí geospatial dat ZABAGED.  
Pipeline kombinuje deterministické Dijkstrovo trasování s reinforcement-learning modelem (PPO/MaskablePPO) nad mřížkovými costmapami.

---

## Struktura projektu

```
config/              # konfigurace nákladových funkcí pro každou zájmovou oblast
data/                # vstupní a zpracovaná data (viz níže)
notebooks/           # Jupyter notebooky – costmapy, vizualizace, inference
src/                 # zdrojové moduly (costmap, Dijkstra, PPO, vizualizace)
ppo_models/          # uložené PPO modely; pro inference je potřeba v4 artefakt
misc/                # poznámky, experimenty, pomocné skripty
data_A1–A4.geojson   # hranice zájmových oblastí
data_B.geojson
requirements.txt
train_ppo.py
```

---

## Požadovaná externí data

Repozitář neobsahuje velké datové soubory. Po naklonování je nutno stáhnout následující soubory a umístit je do odpovídajících adresářů.

### 1. ZABAGED – výsledky prostorových dotazů

| Soubor | Cílové umístění |
|--------|----------------|
| `ZABAGED_RESULTS.gpkg` | `data/raw/ZABAGED_RESULTS.gpkg` |
| `vrstevnice.gpkg` | `data/raw/vrstevnice.gpkg` |

**Zdroj:** ČÚZK – [Základní báze geografických dat ČR (ZABAGED®)](https://geoportal.cuzk.cz/)  
Data lze stáhnout z GeoPortálu ČÚZK po registraci, nebo jsou dostupná přes WFS službu.  
Exportovat je nutno do formátu GeoPackage (`.gpkg`) do adresáře `data/raw/`.

### 2. Elektrické vedení VN110

| Soubor | Cílové umístění |
|--------|----------------|
| `VN110_zabaged_results_elektricke_vedeni.geojson` | `data/layers/VN110_zabaged_results_elektricke_vedeni.geojson` |

**Zdroj:** Výsledek dotazu na vrstvu elektrického vedení z `ZABAGED_RESULTS.gpkg` (lze exportovat z přiloženého `.gpkg` pomocí QGIS nebo `ogr2ogr`).

### 3. PPO model (pro inference)

| Soubor | Cílové umístění |
|--------|----------------|
| `ppo_routing_final.zip` | `ppo_models/coarse300_12km_v4/ppo_routing_final.zip` |

Model je distribuován samostatně (viz releases nebo sdílené úložiště projektu).

---

## Zpracovaná data (generovaná)

Adresář `data/processed/` je prázdný – costmapy se vygenerují spuštěním notebooků:

```
notebooks/costmap_samples_A1.ipynb  →  data/processed/A1/
notebooks/costmap_samples_A2.ipynb  →  data/processed/A2/
notebooks/costmap_samples_A3.ipynb  →  data/processed/A3/
notebooks/costmap_samples_A4.ipynb  →  data/processed/A4/
notebooks/costmap_samples_B.ipynb   →  data/processed/B/
```

---

## Instalace prostředí

Doporučeno Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
# nebo: source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

---

## Spuštění

1. Umístit vstupní data dle tabulky výše.  
2. Spustit příslušný notebook pro generování costmapy (buňka 1).  
3. Spustit buňku 2 pro Dijkstrovo trasování.  
4. Spustit buňky 3–4 pro PPO inferenci (vyžaduje `ppo_routing_final.zip`).

---

## Poznámky

- Costmapy jsou uloženy jako GeoPackage (`.gpkg`) a rastr (`.tif`)
