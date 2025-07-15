# tox_smarts.py  ─  112  alert'ов BRENK (RDKit 2023.09)
from rdkit import Chem

TOX_SMARTS = {
    #  01–10  — реактивные карбонилы, галогениды, изо-/тиоцианаты, азиды, цианаты
    "acid_halide"          : "C(=O)[Cl,Br,I]",                 # R‑CO–Cl/Br/I
    "aldehyde"             : "[CX3H1](=O)[#6]",                # R‑CHO
    "anhydride"            : "O=C(O)C(=O)",                    # (RCO)2O
    "isocyanate"           : "N=C=O",
    "isothiocyanate"       : "N=C=S",
    "cyanate"              : "OC#N",
    "thiocyanate"          : "SC#N",
    "azide"                : "N=[N+]=[N-]",
    "diazo"                : "N=[N+]=N",
    "nitroso"              : "[NX2]=O",

    #  11–20  — нитро‑группы, эпоксиды, азиридины, Michael‑акцепторы, имиды
    "nitro_aliph"          : "[CX3](=O)[NX3](=O)=O",
    "nitro_arom"           : "[$([NX3](=O)=O)](c)",
    "epoxide"              : "[C;r3]1[O;r3][C;r3]1",
    "aziridine"            : "[N;r3]1[C;r3][C;r3]1",
    "aziridine_oxide"      : "C1NC1O",                         # β‑аминированные эпоксиды
    "michael_acceptor"     : "C=CC(=O)",                       # α,β‑E‑unsat. carbonyl
    "quinone"              : "O=1C=CC(=O)C=C1",                # o‑ / p‑quinone
    "imide"                : "C(=O)N(C=O)",                    # succinimide / phthalimide
    "sulfonyl_fluoride"    : "S(=O)(=O)F",
    "chloroacetamide"      : "ClCC(=O)N",

    #  21–30  — α‑галогенкетоны, хлорангидриды сульфокислот, тиоэпоксиды, β‑лактоны
    "alpha_haloketone"     : "[Cl,Br,I]CC(=O)",
    "sulfonyl_chloride"    : "S(=O)(=O)Cl",
    "thioepoxide"          : "[C;r3]1[S;r3][C;r3]1",
    "beta_lactone"         : "C1OC(=O)C1",
    "beta_lactam"          : "C1NC(=O)C1",
    "oxirane_sulfone"      : "C1OC1S(=O)(=O)",
    "chlorooxime"          : "ClC=N[OH]",
    "hydrazine"            : "NN",
    "hydrazone"            : "N=N",
    "semicarbazide"        : "NNC(=O)N",

    #  31–40  — фосфор-/сульфор‑реактивности, силикаты, боронаты, гипергалогены
    "triflate"             : "OS(=O)(=O)C(F)(F)F",
    "sulfone"              : "S(=O)(=O)[#6]",
    "sulfonic_acid"        : "S(=O)(=O)O",
    "phosphonic_acid"      : "P(=O)(O)O",
    "phosphoramide"        : "P(=O)(N)N",
    "boronic_acid"         : "B(O)O",
    "silicate"             : "[Si](O)(O)O",
    "selenooxide"          : "[Se]=O",
    "fluorosulfate"        : "OS(=O)(=O)F",
    "dihaloacetamide"      : "C(Cl)(Cl)C(=O)N",

    #  41–55  — «устойчивые» / трудно метаб. фрагменты, галогенированные алканы
    "trifluoromethyl"      : "C(F)(F)F",
    "pentafluorosulfanyl"  : "S(F)(F)(F)(F)F",
    "perfluoroalkane"      : "C(F)(F)F",      # (обобщённый)
    "tert_butyl"           : "C(C)(C)C",
    "adamantyl"            : "C1C2CC3CC1CC(C2)C3",
    "cubane"               : "C1C2C3C1C4C2C34",
    "norbornane"           : "C12CCCCC1C2",
    "bicyclo_no_bridge"    : "C1CCC2CCCCC12",
    "polycyclic_arom"      : "c1ccc2ccc3cccc4ccc1c2c34",
    "alkyl_chloride"       : "CCl",

    #  56–68  — тиолы, тиоэфиры, сульфиды, соли металлов (Hg, As, Pb, Sn)
    "thiol"                : "[SX2H]",
    "disulfide"            : "SS",
    "sulfide"              : "[SX2]([#6])[#6]",
    "thioether_alkenyl"    : "CS[CX3]=[CX3]",
    "arsenic"              : "[As]",
    "mercury"              : "[Hg]",
    "lead"                 : "[Pb]",
    "tin"                  : "[Sn]",
    "selenide"             : "[Se][#6]",
    "telluride"            : "[Te][#6]",
    "thiosulfonate"        : "S(=O)(=O)S",

    #  69–80  — нитрофуран, нитротриазол, бензофуран, бензотиофен, индол
    "nitrofuran"           : "c1oc([NX3](=O)=O)cc1",
    "nitroimidazole"       : "c1ncc([NX3](=O)=O)n1",
    "nitrothiazole"        : "c1nc([NX3](=O)=O)s1",
    "benzofuran"           : "c1ccc2occc2c1",
    "benzothiophene"       : "c1ccc2sccc2c1",
    "indole"               : "c1ccc2[nH]ccc2c1",
    "carbazole"            : "c1ccc2c(c1)[nH]c3ccccc23",
    "azepine"              : "c1ccccn1",
    "fluoro_anilide"       : "c1ccc(cc1)NC(=O)C(F)(F)F",
    "para_anilide"         : "c1ccc(cc1)NC(=O)c2ccc(cc2)",

    #  81–92  — фенолы, резорцины, гидрохинон, катехол, хиноны, пирокатехины
    "phenol"               : "c1ccc(cc1)O",
    "catechol"             : "c1cc(c(c(c1)O)O)",
    "resorcinol"           : "c1cc(cc(c1)O)O",
    "hydroquinone"         : "c1ccc(c(c1)O)O",
    "ortho_quinone"        : "O=1C=CC(=O)C=C1",
    "para_quinone"         : "O=1CC(=O)C=CC1",
    "aminophenol"          : "c1ccc(cc1)N",
    "aniline"              : "c1ccc(cc1)N",
    "p_anisidine"          : "COc1ccc(cc1)N",
    "p_aminobenzoic_acid"  : "NC1=CC=C(C=C1)C(=O)O",
    "arylhydrazine"        : "c1ccc(cc1)NN",

    #  93–104 — углеводородные полициклы, диазо‑/азо‑соединения, барбитураты
    "azo"                  : "N=N",
    "diazonium"            : "[N+](=N)[O-]",
    "diazo_methane"        : "N=[N+]=[C-]",
    "barbituric_acid"      : "C1(=O)NC(=O)NC(=O)N1",
    "dihydropyridine"      : "C1=NC=CCC1",
    "benzodiazepine"       : "N1C(=O)c2ccccc2C(N1)N",
    "bicyclic_small_ring"  : "[r3,r4&r5&R]",
    "spiro_small"          : "[CX4]1(CC1)C",
    "cubyl_halide"         : "C12C3C4C1C5C2C3C45[Cl,Br,I]",
    "polyaryl_pyridine"    : "c1ccc(cc1)c2ncccc2",

    # 105–112 — галоген‑тиофосфаты, гидразид‑тиоэфиры, пара‑гидроксил, др.
    "chloro_oxime"         : "ClC=NO",
    "bromo_oxime"          : "BrC=NO",
    "para_hydroxylation"   : "c1ccc(cc1)O",
    "chlorosulfonate"      : "ClS(=O)(=O)O",
    "phosphoramide_mustard": "N[P](=O)(Cl)Cl",
    "nitrogen_mustard"     : "NCCCl",
    "thiocarbonyl"         : "C=S",
    "thiourea"             : "N=C(S)N",
    "triazene"             : "N=N[CX3]",
    "selone"               : "[Se]=C",
    "tellone"              : "[Te]=C",
    "thioanhydride"        : "C(=S)SC(=O)"       # mixed thio‑/oxo‑anhydride
}

# RDKit-паттерны для ускорения поиска
TOX_PATTERNS = {name: Chem.MolFromSmarts(sma) for name, sma in TOX_SMARTS.items()}
