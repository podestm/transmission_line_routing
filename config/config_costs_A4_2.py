# ===========================================================
# Konfigurace vrstev hybridniho cost fieldu
# ===========================================================

# === Parametry grafu ===
GRAPH_CONFIG = {
    "crs": 5514,
}

# === Parametry vysek a sklonu ===
ELEVATION_CONFIG = {
    "contour_path": "data/raw/vrstevnice.gpkg",
    "height_field": None,
    "height_field_candidates": ["VYSKA", "vyska__m_", "vyska_m", "height", "elevation", "z"],
    "max_slope_deg": 45.0,
    "penalty_from_deg": 20.0,
    "penalty_max_multiplier": 2.0,
}

# === Jednotna konfigurace vrstev (cost + pravidla) ===
#
# Poznamka k inward gradientu:
# - plati jen pro vybrane chranene vrstvy
# - gradient je uvnitr polygonu (od okraje dovnitr), ne ven
# - "distance" = 800 znamena pas 800 m od hranice smerem dovnitr
#
# Vrstvy jsou rozdeleny do 5 skupin:
#   1) Pruchozi plochy (nizky cost)
#   2) Pruchozi liniove prvky s buffer exkluzi (silnice, zeleznice)
#   3) Vodni plochy (vyssi cost, buffer exkluze)
#   4) Exkluzni zastavba a infrastruktura (cost=inf)
#   5) Chranena uzemi (cost=inf, volitelny inward gradient)

LAYER_DEFINITIONS = {

    # -----------------------------------------------------------
    # 1) PRUCHOZI PLOCHY â€” nizky cost, lze umistit uzly
    # -----------------------------------------------------------
    "OrnaPudaAOstatniDaleNespecifikovanePlochy": {
        "cost": 1.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "TrvalyTravniPorost": {
        "cost": 1.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "LesniPudaSKrovinatymPorostem": {
        "cost": 1.8,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "OstatniPlochaVSidlech": {
        "cost": 10.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "ParkovisteOdpocivka": {
        "cost": 5.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "OkrasnaZahradaPark": {
        "cost": 10.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "LesniPudaSKosodrevinou": {
        "cost": 2.6,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "Vinice": {
        "cost": 3.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "LesniPudaSeStromy": {
        "cost": 8.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "BazinaMocal": {
        "cost": 10.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "Raseliniste": {
        "cost": 5.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "SesuvPudySut": {
        "cost": 6.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },
    "SkalniUtvary": {
        "cost": 10.0,
        "can_place_nodes": True,
        "buffer_exclusion": 0,
    },

    # -----------------------------------------------------------
    # 2) LINIOVE PRVKY S BUFFER EXKLUZI â€” pruchozi, ale nelze
    #    umistit uzly; buffer kolem osy vytvari exkluzni zonu
    # -----------------------------------------------------------
    "Ulice": {
        "cost": 5.0,
        "can_place_nodes": False,
        "buffer_exclusion": 40,
        "can_cross": True,
    },
    "SilniceDalnice": {
        "cost": 10.0,
        "can_place_nodes": False,
        "buffer_exclusion": 30,
        "can_cross": True,
    },
    "ZeleznicniTrat": {
        "cost": 10.0,
        "can_place_nodes": False,
        "buffer_exclusion": 30,
        "can_cross": True,
    },

    # -----------------------------------------------------------
    # 3) VODNI PLOCHY â€” vyssi cost, buffer exkluze
    # -----------------------------------------------------------
    "VodniPlocha": {
        "cost": 10.0,
        "can_place_nodes": False,
        "buffer_exclusion": 30,
        "can_cross": True,
    },

    # -----------------------------------------------------------
    # 4) EXKLUZNI ZASTAVBA A INFRASTRUKTURA â€” cost=inf, nelze
    #    krizovat ani umistit uzly
    # -----------------------------------------------------------
    "ArealUceloveZastavby": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 20,
        "can_cross": False,
    },
    "BudovaJednotlivaNeboBlokBudov": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 20,
        "can_cross": False,
    },
    "OvocnySadZahrada": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 20,
        "can_cross": False,
    },
    "Letiste": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 200,
        "can_cross": False,
    },
    "ObvodLetistniDrahy": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 200,
        "can_cross": False,
    },
    "RozvodnaTransformovna": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 20,
        "can_cross": False,
    },
    "Elektrarna": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 20,
        "can_cross": False,
    },
    "Skladka": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 20,
        "can_cross": False,
    },
    "PovrchovaTezbaLom": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 20,
        "can_cross": False,
    },
    "Tribuna": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 200,
        "can_cross": False,
    },
    "Hrad": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 200,
        "can_cross": False,
    },
    "Zamek": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 200,
        "can_cross": False,
    },

    # -----------------------------------------------------------
    # 5) CHRANENA UZEMI â€” cost=inf, volitelny inward gradient
    #    (gradient = postupny prechod od okraje dovnitr)
    # -----------------------------------------------------------
    "MaloplosneZvlasteChraneneUzemi": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 0,
        "can_cross": False,
    },
    "VelkoplosneZvlasteChraneneUzemi": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 0,
        "can_cross": False,
        "inward_gradient": {
            "distance": 800.0,
            "cost_outer": 1.0,
            "cost_inner": 4.0,
        },
    },
    "EvropskyVyznamnaLokalita": {
        "cost": float("inf"),
        "can_place_nodes": False,
        "buffer_exclusion": 0,
        "can_cross": False,
        "inward_gradient": {
            "distance": 800.0,
            "cost_outer": 1.0,
            "cost_inner": 4.0,
        },
    },
}

# === Odvozene mapy pro zpetnou kompatibilitu ===
LAYER_COSTS = {name: cfg["cost"] for name, cfg in LAYER_DEFINITIONS.items()}

LAYER_RULES = {
    name: {k: v for k, v in cfg.items() if k != "cost"}
    for name, cfg in LAYER_DEFINITIONS.items()
}

