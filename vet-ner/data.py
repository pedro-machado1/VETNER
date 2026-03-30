# data.py
# Dataset de NER clínico veterinário com anotação BIO
# Entidades: DOENCA, SINTOMA, MEDICAMENTO, ESPECIE, TRATAMENTO
# 150 sentenças anotadas manualmente

import json
import random
from pathlib import Path
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

LABELS = [
    "O",
    "B-DOENCA", "I-DOENCA",
    "B-SINTOMA", "I-SINTOMA",
    "B-MEDICAMENTO", "I-MEDICAMENTO",
    "B-ESPECIE", "I-ESPECIE",
    "B-TRATAMENTO", "I-TRATAMENTO",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

RAW_DATA = [
    # ── DERMATOLOGIA ─────────────────────────────────────────────
    [("O","O"),("cão","B-ESPECIE"),("apresentou","O"),("tosse","B-SINTOMA"),
     ("persistente","I-SINTOMA"),("e","O"),("secreção","B-SINTOMA"),("nasal","I-SINTOMA"),
     ("sugestivos","O"),("de","O"),("traqueobronquite","B-DOENCA"),("infecciosa","I-DOENCA"),(".",  "O")],

    [("A","O"),("gata","B-ESPECIE"),("estava","O"),("letárgica","B-SINTOMA"),(",","O"),
     ("sem","O"),("apetite","B-SINTOMA"),("e","O"),("com","O"),("vômitos","B-SINTOMA"),
     ("frequentes","I-SINTOMA"),(".",  "O")],

    [("Diagnosticou-se","O"),("diabetes","B-DOENCA"),("mellitus","I-DOENCA"),("no","O"),
     ("felino","B-ESPECIE"),("com","O"),("poliúria","B-SINTOMA"),(",","O"),
     ("polidipsia","B-SINTOMA"),("e","O"),("perda","B-SINTOMA"),("de","I-SINTOMA"),
     ("peso","I-SINTOMA"),(".",  "O")],

    [("O","O"),("tratamento","O"),("com","O"),("oclacitinib","B-MEDICAMENTO"),
     ("reduziu","O"),("o","O"),("prurido","B-SINTOMA"),("na","O"),
     ("dermatite","B-DOENCA"),("atópica","I-DOENCA"),("canina","O"),(".",  "O")],

    [("Prescreveu-se","O"),("ivermectina","B-MEDICAMENTO"),("para","O"),("tratar","O"),
     ("a","O"),("sarna","B-DOENCA"),("sarcóptica","I-DOENCA"),("no","O"),
     ("cão","B-ESPECIE"),(".",  "O")],

    [("A","O"),("hemilaminectomia","B-TRATAMENTO"),("foi","O"),("indicada","O"),
     ("após","O"),("diagnóstico","O"),("de","O"),("DDIV","B-DOENCA"),("grau","O"),
     ("IV","O"),("no","O"),("Dachshund","B-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("idoso","O"),("apresentava","O"),
     ("perda","B-SINTOMA"),("de","I-SINTOMA"),("peso","I-SINTOMA"),(",","O"),
     ("polifagia","B-SINTOMA"),("e","O"),("taquicardia","B-SINTOMA"),
     ("compatíveis","O"),("com","O"),("hipertireoidismo","B-DOENCA"),(".",  "O")],

    [("Administrou-se","O"),("metimazol","B-MEDICAMENTO"),("como","O"),
     ("tratamento","B-TRATAMENTO"),("inicial","I-TRATAMENTO"),("do","O"),
     ("hipertireoidismo","B-DOENCA"),("felino","B-ESPECIE"),(".",  "O")],

    [("O","O"),("linfoma","B-DOENCA"),("multicêntrico","I-DOENCA"),("canino","O"),
     ("foi","O"),("tratado","O"),("com","O"),("protocolo","O"),("CHOP","B-TRATAMENTO"),
     ("contendo","O"),("ciclofosfamida","B-MEDICAMENTO"),(",","O"),
     ("doxorrubicina","B-MEDICAMENTO"),("e","O"),("vincristina","B-MEDICAMENTO"),(".",  "O")],

    [("Paciente","O"),("canino","B-ESPECIE"),("com","O"),("eritema","B-SINTOMA"),(",","O"),
     ("escoriações","B-SINTOMA"),("e","O"),("coceira","B-SINTOMA"),("intensa","I-SINTOMA"),
     ("nas","O"),("patas","O"),("—","O"),("suspeita","O"),("de","O"),
     ("dermatite","B-DOENCA"),("atópica","I-DOENCA"),(".",  "O")],

    [("A","O"),("insuficiência","B-DOENCA"),("renal","I-DOENCA"),("crônica","I-DOENCA"),
     ("é","O"),("muito","O"),("prevalente","O"),("em","O"),("gatos","B-ESPECIE"),
     ("acima","O"),("de","O"),("15","O"),("anos","O"),(".",  "O")],

    [("O","O"),("benazepril","B-MEDICAMENTO"),("foi","O"),("prescrito","O"),
     ("para","O"),("reduzir","O"),("a","O"),("proteinúria","B-SINTOMA"),
     ("na","O"),("DRC","B-DOENCA"),("felina","O"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("apresentou","O"),("distensão","B-SINTOMA"),
     ("abdominal","I-SINTOMA"),("aguda","O"),("e","O"),("salivação","B-SINTOMA"),
     ("excessiva","I-SINTOMA"),(",","O"),("sugerindo","O"),
     ("torção","B-DOENCA"),("gástrica","I-DOENCA"),(".",  "O")],

    [("A","O"),("cirurgia","B-TRATAMENTO"),("de","I-TRATAMENTO"),
     ("destorção","I-TRATAMENTO"),("foi","O"),("realizada","O"),("com","O"),
     ("urgência","O"),("no","O"),("Pastor","B-ESPECIE"),("Alemão","I-ESPECIE"),(".",  "O")],

    [("Fluralaner","B-MEDICAMENTO"),("mostrou-se","O"),("eficaz","O"),
     ("no","O"),("controle","O"),("de","O"),("pulgas","B-DOENCA"),("e","O"),
     ("carrapatos","B-DOENCA"),("em","O"),("cães","B-ESPECIE"),("e","O"),
     ("gatos","B-ESPECIE"),(".",  "O")],

    [("O","O"),("mastocitoma","B-DOENCA"),("grau","O"),("II","O"),("foi","O"),
     ("tratado","O"),("com","O"),("vimblastina","B-MEDICAMENTO"),("e","O"),
     ("prednisona","B-MEDICAMENTO"),("após","O"),("excisão","B-TRATAMENTO"),
     ("cirúrgica","I-TRATAMENTO"),(".",  "O")],

    [("O","O"),("coelho","B-ESPECIE"),("apresentou","O"),("head","B-SINTOMA"),
     ("tilt","I-SINTOMA"),("e","O"),("nistagmo","B-SINTOMA"),("associados","O"),
     ("a","O"),("otite","B-DOENCA"),("média","I-DOENCA"),(".",  "O")],

    [("A","O"),("imunoterapia","B-TRATAMENTO"),("alérgeno-específica","I-TRATAMENTO"),
     ("é","O"),("opção","O"),("para","O"),("controle","O"),("da","O"),
     ("dermatite","B-DOENCA"),("atópica","I-DOENCA"),("em","O"),("longo","O"),("prazo","O"),(".",  "O")],

    [("O","O"),("toceranib","B-MEDICAMENTO"),("inibe","O"),("tirosina","O"),
     ("quinase","O"),("e","O"),("é","O"),("usado","O"),("em","O"),
     ("mastocitomas","B-DOENCA"),("com","O"),("mutação","O"),("do","O"),("c-KIT","O"),(".",  "O")],

    [("Gatos","B-ESPECIE"),("necessitam","O"),("de","O"),("taurina","B-MEDICAMENTO"),
     ("na","O"),("dieta","O"),("para","O"),("prevenir","O"),
     ("cardiomiopatia","B-DOENCA"),("dilatada","I-DOENCA"),(".",  "O")],

    # ── CLÍNICA MÉDICA ────────────────────────────────────────────
    [("O","O"),("Beagle","B-ESPECIE"),("apresentou","O"),("febre","B-SINTOMA"),(",","O"),
     ("linfadenopatia","B-SINTOMA"),("generalizada","I-SINTOMA"),("e","O"),("perda","B-SINTOMA"),
     ("de","I-SINTOMA"),("peso","I-SINTOMA"),("compatíveis","O"),("com","O"),
     ("leishmaniose","B-DOENCA"),("visceral","I-DOENCA"),(".",  "O")],

    [("A","O"),("erliquiose","B-DOENCA"),("canina","O"),("foi","O"),("confirmada","O"),
     ("e","O"),("tratada","O"),("com","O"),("doxiciclina","B-MEDICAMENTO"),
     ("por","O"),("28","O"),("dias","O"),(".",  "O")],

    [("O","O"),("Golden","B-ESPECIE"),("Retriever","I-ESPECIE"),("apresentou","O"),
     ("claudicação","B-SINTOMA"),("e","O"),("dor","B-SINTOMA"),("articular","I-SINTOMA"),
     ("sugestivos","O"),("de","O"),("displasia","B-DOENCA"),("coxofemoral","I-DOENCA"),(".",  "O")],

    [("Felino","B-ESPECIE"),("com","O"),("hematúria","B-SINTOMA"),(",","O"),
     ("disúria","B-SINTOMA"),("e","O"),("estrangúria","B-SINTOMA"),
     ("foi","O"),("diagnosticado","O"),("com","O"),("cistite","B-DOENCA"),
     ("idiopática","I-DOENCA"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("recebeu","O"),("transfusão","B-TRATAMENTO"),
     ("sanguínea","I-TRATAMENTO"),("devido","O"),("à","O"),
     ("anemia","B-DOENCA"),("hemolítica","I-DOENCA"),("imunomediada","I-DOENCA"),(".",  "O")],

    [("Prescreveu-se","O"),("prednisolona","B-MEDICAMENTO"),("para","O"),
     ("o","O"),("controle","O"),("da","O"),("anemia","B-DOENCA"),
     ("hemolítica","I-DOENCA"),("imunomediada","I-DOENCA"),("no","O"),
     ("Cocker","B-ESPECIE"),("Spaniel","I-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("apresentou","O"),("icterícia","B-SINTOMA"),(",","O"),
     ("anorexia","B-SINTOMA"),("e","O"),("hepatomegalia","B-SINTOMA"),
     ("compatíveis","O"),("com","O"),("lipidose","B-DOENCA"),("hepática","I-DOENCA"),(".",  "O")],

    [("A","O"),("nutrição","B-TRATAMENTO"),("enteral","I-TRATAMENTO"),("via","O"),
     ("sonda","O"),("nasoesofágica","O"),("foi","O"),("indicada","O"),
     ("para","O"),("o","O"),("gato","B-ESPECIE"),("com","O"),
     ("lipidose","B-DOENCA"),("hepática","I-DOENCA"),(".",  "O")],

    [("O","O"),("Rottweiler","B-ESPECIE"),("foi","O"),("internado","O"),
     ("com","O"),("convulsões","B-SINTOMA"),("e","O"),("ataxia","B-SINTOMA"),
     ("após","O"),("diagnóstico","O"),("de","O"),("cinomose","B-DOENCA"),(".",  "O")],

    [("O","O"),("parvovírus","B-DOENCA"),("canino","O"),("causou","O"),
     ("vômito","B-SINTOMA"),("intenso","I-SINTOMA"),(",","O"),("diarreia","B-SINTOMA"),
     ("hemorrágica","I-SINTOMA"),("e","O"),("leucopenia","B-SINTOMA"),
     ("no","O"),("filhote","B-ESPECIE"),(".",  "O")],

    [("O","O"),("tratamento","O"),("do","O"),("parvovírus","B-DOENCA"),("incluiu","O"),
     ("fluidoterapia","B-TRATAMENTO"),(",","O"),("antieméticos","B-MEDICAMENTO"),
     ("e","O"),("antibioticoterapia","B-TRATAMENTO"),("de","I-TRATAMENTO"),
     ("suporte","I-TRATAMENTO"),(".",  "O")],

    [("O","O"),("Poodle","B-ESPECIE"),("com","O"),("síncope","B-SINTOMA"),
     ("e","O"),("intolerância","B-SINTOMA"),("ao","O"),("exercício","I-SINTOMA"),
     ("foi","O"),("diagnosticado","O"),("com","O"),
     ("cardiomiopatia","B-DOENCA"),("dilatada","I-DOENCA"),(".",  "O")],

    [("Enalapril","B-MEDICAMENTO"),("e","O"),("furosemida","B-MEDICAMENTO"),
     ("foram","O"),("prescritos","O"),("para","O"),("o","O"),
     ("manejo","B-TRATAMENTO"),("da","I-TRATAMENTO"),("insuficiência","B-DOENCA"),
     ("cardíaca","I-DOENCA"),("congestiva","I-DOENCA"),(".",  "O")],

    [("O","O"),("Shih","B-ESPECIE"),("Tzu","I-ESPECIE"),("apresentou","O"),
     ("prurido","B-SINTOMA"),("intenso","I-SINTOMA"),(",","O"),("pele","O"),
     ("ressecada","B-SINTOMA"),("e","O"),("perda","B-SINTOMA"),("de","I-SINTOMA"),
     ("pelo","I-SINTOMA"),(".",  "O")],

    [("A","O"),("demodiciose","B-DOENCA"),("generalizada","I-DOENCA"),("no","O"),
     ("Shar","B-ESPECIE"),("Pei","I-ESPECIE"),("foi","O"),("tratada","O"),
     ("com","O"),("afoxolaner","B-MEDICAMENTO"),("mensal","O"),(".",  "O")],

    [("O","O"),("felino","B-ESPECIE"),("apresentou","O"),("rinite","B-DOENCA"),
     ("crónica","I-DOENCA"),("com","O"),("espirros","B-SINTOMA"),("frequentes","I-SINTOMA"),
     ("e","O"),("secreção","B-SINTOMA"),("mucopurulenta","I-SINTOMA"),(".",  "O")],

    [("Amoxicilina","B-MEDICAMENTO"),("com","O"),("clavulanato","B-MEDICAMENTO"),
     ("foi","O"),("prescrita","O"),("para","O"),("infecção","B-DOENCA"),
     ("bacteriana","I-DOENCA"),("secundária","I-DOENCA"),("no","O"),
     ("gato","B-ESPECIE"),(".",  "O")],

    [("O","O"),("Border","B-ESPECIE"),("Collie","I-ESPECIE"),("apresentou","O"),
     ("tremores","B-SINTOMA"),(",","O"),("desorientação","B-SINTOMA"),("e","O"),
     ("midríase","B-SINTOMA"),("após","O"),("ingestão","O"),("de","O"),
     ("ivermectina","B-MEDICAMENTO"),(".",  "O")],

    [("O","O"),("envenenamento","B-DOENCA"),("por","I-DOENCA"),("ivermectina","B-MEDICAMENTO"),
     ("em","O"),("cães","B-ESPECIE"),("MDR1","O"),("requer","O"),
     ("suporte","B-TRATAMENTO"),("neurológico","I-TRATAMENTO"),("intensivo","I-TRATAMENTO"),(".",  "O")],

    [("O","O"),("Labrador","B-ESPECIE"),("foi","O"),("submetido","O"),
     ("a","O"),("artroscopia","B-TRATAMENTO"),("para","O"),("tratar","O"),
     ("a","O"),("ruptura","B-DOENCA"),("do","I-DOENCA"),("ligamento","I-DOENCA"),
     ("cruzado","I-DOENCA"),("cranial","I-DOENCA"),(".",  "O")],

    [("A","O"),("fisioterapia","B-TRATAMENTO"),("e","O"),("hidroterapia","B-TRATAMENTO"),
     ("foram","O"),("indicadas","O"),("no","O"),("pós-operatório","O"),
     ("da","O"),("TPLO","B-TRATAMENTO"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    # ── ONCOLOGIA ────────────────────────────────────────────────
    [("O","O"),("osteossarcoma","B-DOENCA"),("apendicular","I-DOENCA"),("foi","O"),
     ("diagnosticado","O"),("no","O"),("Dogue","B-ESPECIE"),("de","I-ESPECIE"),
     ("Bordeaux","I-ESPECIE"),("com","O"),("claudicação","B-SINTOMA"),
     ("progressiva","I-SINTOMA"),(".",  "O")],

    [("A","O"),("amputação","B-TRATAMENTO"),("do","I-TRATAMENTO"),("membro","I-TRATAMENTO"),
     ("seguida","O"),("de","O"),("quimioterapia","B-TRATAMENTO"),("com","O"),
     ("carboplatina","B-MEDICAMENTO"),("foi","O"),("recomendada","O"),
     ("para","O"),("o","O"),("osteossarcoma","B-DOENCA"),(".",  "O")],

    [("O","O"),("carcinoma","B-DOENCA"),("mamário","I-DOENCA"),("foi","O"),
     ("identificado","O"),("na","O"),("cadela","B-ESPECIE"),("durante","O"),
     ("exame","O"),("de","O"),("rotina","O"),(".",  "O")],

    [("A","O"),("mastectomia","B-TRATAMENTO"),("unilateral","I-TRATAMENTO"),("foi","O"),
     ("realizada","O"),("para","O"),("remoção","O"),("do","O"),
     ("carcinoma","B-DOENCA"),("mamário","I-DOENCA"),("na","O"),
     ("cadela","B-ESPECIE"),(".",  "O")],

    [("O","O"),("linfoma","B-DOENCA"),("mediastinal","I-DOENCA"),("causou","O"),
     ("dispneia","B-SINTOMA"),("e","O"),("efusão","B-SINTOMA"),("pleural","I-SINTOMA"),
     ("no","O"),("gato","B-ESPECIE"),(".",  "O")],

    [("A","O"),("toracocentese","B-TRATAMENTO"),("foi","O"),("realizada","O"),
     ("para","O"),("aliviar","O"),("a","O"),("dispneia","B-SINTOMA"),
     ("causada","O"),("pela","O"),("efusão","B-SINTOMA"),("pleural","I-SINTOMA"),(".",  "O")],

    [("O","O"),("hemangiossarcoma","B-DOENCA"),("esplênico","I-DOENCA"),("foi","O"),
     ("descoberto","O"),("após","O"),("o","O"),("cão","B-ESPECIE"),
     ("apresentar","O"),("colapso","B-SINTOMA"),("súbito","I-SINTOMA"),(".",  "O")],

    [("A","O"),("esplenectomia","B-TRATAMENTO"),("de","I-TRATAMENTO"),("emergência","I-TRATAMENTO"),
     ("foi","O"),("realizada","O"),("no","O"),("cão","B-ESPECIE"),
     ("com","O"),("hemangiossarcoma","B-DOENCA"),("esplênico","I-DOENCA"),(".",  "O")],

    [("O","O"),("fibrossarcoma","B-DOENCA"),("pós-vacinal","I-DOENCA"),("é","O"),
     ("uma","O"),("complicação","O"),("rara","O"),("mas","O"),("grave","O"),
     ("em","O"),("gatos","B-ESPECIE"),(".",  "O")],

    [("A","O"),("excisão","B-TRATAMENTO"),("ampla","I-TRATAMENTO"),("com","O"),
     ("margens","O"),("de","O"),("segurança","O"),("é","O"),("essencial","O"),
     ("no","O"),("fibrossarcoma","B-DOENCA"),("felino","O"),(".",  "O")],

    # ── NEUROLOGIA ───────────────────────────────────────────────
    [("O","O"),("Cavalier","B-ESPECIE"),("King","I-ESPECIE"),("Charles","I-ESPECIE"),
     ("apresentou","O"),("siringomielia","B-DOENCA"),("com","O"),
     ("coceira","B-SINTOMA"),("compulsiva","I-SINTOMA"),("no","O"),("pescoço","O"),(".",  "O")],

    [("A","O"),("gabapentina","B-MEDICAMENTO"),("foi","O"),("prescrita","O"),
     ("para","O"),("controle","O"),("da","O"),("dor","B-SINTOMA"),
     ("neuropática","I-SINTOMA"),("na","O"),("siringomielia","B-DOENCA"),(".",  "O")],

    [("O","O"),("Boxer","B-ESPECIE"),("apresentou","O"),("convulsões","B-SINTOMA"),
     ("focais","I-SINTOMA"),("recorrentes","I-SINTOMA"),("compatíveis","O"),
     ("com","O"),("epilepsia","B-DOENCA"),("idiopática","I-DOENCA"),(".",  "O")],

    [("Fenobarbital","B-MEDICAMENTO"),("foi","O"),("iniciado","O"),("para","O"),
     ("controle","B-TRATAMENTO"),("das","I-TRATAMENTO"),("crises","I-TRATAMENTO"),
     ("epilépticas","I-TRATAMENTO"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("apresentou","O"),("paralisia","B-SINTOMA"),
     ("dos","I-SINTOMA"),("membros","I-SINTOMA"),("posteriores","I-SINTOMA"),
     ("após","O"),("trauma","O"),("vertebral","O"),(".",  "O")],

    [("A","O"),("ressonância","B-TRATAMENTO"),("magnética","I-TRATAMENTO"),
     ("confirmou","O"),("a","O"),("compressão","B-DOENCA"),("medular","I-DOENCA"),
     ("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("Retriever","B-ESPECIE"),("apresentou","O"),("fraqueza","B-SINTOMA"),
     ("progressiva","I-SINTOMA"),("e","O"),("regurgitação","B-SINTOMA"),
     ("associadas","O"),("a","O"),("miastenia","B-DOENCA"),("gravis","I-DOENCA"),(".",  "O")],

    [("Piridostigmina","B-MEDICAMENTO"),("foi","O"),("indicada","O"),("para","O"),
     ("o","O"),("tratamento","B-TRATAMENTO"),("da","I-TRATAMENTO"),
     ("miastenia","B-DOENCA"),("gravis","I-DOENCA"),("no","O"),
     ("Retriever","B-ESPECIE"),(".",  "O")],

    # ── ENDOCRINOLOGIA ───────────────────────────────────────────
    [("A","O"),("hiperadrenocorticismo","B-DOENCA"),("foi","O"),("diagnosticada","O"),
     ("no","O"),("cão","B-ESPECIE"),("com","O"),("poliúria","B-SINTOMA"),(",","O"),
     ("polidipsia","B-SINTOMA"),("e","O"),("abdômen","B-SINTOMA"),("pendular","I-SINTOMA"),(".",  "O")],

    [("Trilostano","B-MEDICAMENTO"),("foi","O"),("prescrito","O"),("para","O"),
     ("controle","O"),("da","O"),("doença","B-DOENCA"),("de","I-DOENCA"),
     ("Cushing","I-DOENCA"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("hipotireoidismo","B-DOENCA"),("foi","O"),("identificado","O"),
     ("no","O"),("Doberman","B-ESPECIE"),("com","O"),("obesidade","B-SINTOMA"),(",","O"),
     ("letargia","B-SINTOMA"),("e","O"),("alopecia","B-SINTOMA"),("bilateral","I-SINTOMA"),(".",  "O")],

    [("Levotiroxina","B-MEDICAMENTO"),("sódica","O"),("foi","O"),("administrada","O"),
     ("diariamente","O"),("para","O"),("o","O"),("cão","B-ESPECIE"),
     ("com","O"),("hipotireoidismo","B-DOENCA"),(".",  "O")],

    [("O","O"),("hipoadrenocorticismo","B-DOENCA"),("se","O"),("manifestou","O"),
     ("com","O"),("fraqueza","B-SINTOMA"),(",","O"),("vômito","B-SINTOMA"),
     ("e","O"),("hipoglicemia","B-SINTOMA"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("A","O"),("suplementação","B-TRATAMENTO"),("com","I-TRATAMENTO"),
     ("acetato","B-MEDICAMENTO"),("de","I-MEDICAMENTO"),("fludrocortisona","I-MEDICAMENTO"),
     ("foi","O"),("iniciada","O"),("para","O"),("o","O"),
     ("hipoadrenocorticismo","B-DOENCA"),(".",  "O")],

    [("O","O"),("insulinoma","B-DOENCA"),("causou","O"),("episódios","O"),
     ("de","O"),("hipoglicemia","B-SINTOMA"),("e","O"),("convulsões","B-SINTOMA"),
     ("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("A","O"),("pancreatectomia","B-TRATAMENTO"),("parcial","I-TRATAMENTO"),
     ("foi","O"),("realizada","O"),("para","O"),("remoção","O"),
     ("do","O"),("insulinoma","B-DOENCA"),(".",  "O")],

    # ── INFECTOLOGIA ─────────────────────────────────────────────
    [("O","O"),("cão","B-ESPECIE"),("apresentou","O"),("febre","B-SINTOMA"),(",","O"),
     ("epistaxe","B-SINTOMA"),("e","O"),("trombocitopenia","B-SINTOMA"),
     ("compatíveis","O"),("com","O"),("erliquiose","B-DOENCA"),("monocítica","I-DOENCA"),(".",  "O")],

    [("A","O"),("babesiose","B-DOENCA"),("foi","O"),("tratada","O"),("com","O"),
     ("dipropionato","B-MEDICAMENTO"),("de","I-MEDICAMENTO"),("imidocarb","I-MEDICAMENTO"),
     ("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("FIV","O"),("positivo","O"),("apresentou","O"),
     ("estomatite","B-DOENCA"),("crônica","I-DOENCA"),("com","O"),
     ("ulcerações","B-SINTOMA"),("orais","I-SINTOMA"),("extensas","I-SINTOMA"),(".",  "O")],

    [("A","O"),("extração","B-TRATAMENTO"),("dentária","I-TRATAMENTO"),("total","I-TRATAMENTO"),
     ("foi","O"),("realizada","O"),("no","O"),("gato","B-ESPECIE"),
     ("com","O"),("estomatite","B-DOENCA"),("refratária","I-DOENCA"),(".",  "O")],

    [("O","O"),("toxoplasma","B-DOENCA"),("gondii","I-DOENCA"),("causou","O"),
     ("uveíte","B-SINTOMA"),("e","O"),("coriorretinite","B-SINTOMA"),
     ("no","O"),("felino","B-ESPECIE"),(".",  "O")],

    [("Clindamicina","B-MEDICAMENTO"),("foi","O"),("prescrita","O"),("para","O"),
     ("o","O"),("tratamento","B-TRATAMENTO"),("da","I-TRATAMENTO"),
     ("toxoplasmose","B-DOENCA"),("felina","O"),(".",  "O")],

    [("A","O"),("leptospirose","B-DOENCA"),("causou","O"),("icterícia","B-SINTOMA"),(",","O"),
     ("insuficiência","B-DOENCA"),("renal","I-DOENCA"),("aguda","I-DOENCA"),
     ("e","O"),("hemorragia","B-SINTOMA"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("Penicilina","B-MEDICAMENTO"),("G","I-MEDICAMENTO"),("foi","O"),("administrada","O"),
     ("para","O"),("o","O"),("tratamento","B-TRATAMENTO"),("da","I-TRATAMENTO"),
     ("leptospirose","B-DOENCA"),("aguda","O"),(".",  "O")],

    # ── MEDICINA DE ANIMAIS SILVESTRES ───────────────────────────
    [("O","O"),("papagaio","B-ESPECIE"),("apresentou","O"),("dispneia","B-SINTOMA"),(",","O"),
     ("secreção","B-SINTOMA"),("nasal","I-SINTOMA"),("e","O"),("penas","O"),
     ("arrepiadas","B-SINTOMA"),(".",  "O")],

    [("O","O"),("diagnóstico","O"),("de","O"),("aspergilose","B-DOENCA"),
     ("pulmonar","I-DOENCA"),("foi","O"),("confirmado","O"),("no","O"),
     ("psitacídeo","B-ESPECIE"),("por","O"),("cultura","O"),("fúngica","O"),(".",  "O")],

    [("Voriconazol","B-MEDICAMENTO"),("foi","O"),("utilizado","O"),("para","O"),
     ("tratar","O"),("a","O"),("aspergilose","B-DOENCA"),("no","O"),
     ("papagaio","B-ESPECIE"),(".",  "O")],

    [("A","O"),("tartaruga","B-ESPECIE"),("apresentou","O"),("anorexia","B-SINTOMA"),
     ("e","O"),("descarga","B-SINTOMA"),("nasal","I-SINTOMA"),("sugestivas","O"),
     ("de","O"),("pneumonia","B-DOENCA"),("bacteriana","I-DOENCA"),(".",  "O")],

    [("O","O"),("furão","B-ESPECIE"),("com","O"),("alopecia","B-SINTOMA"),
     ("bilateral","I-SINTOMA"),("e","O"),("prurido","B-SINTOMA"),("foi","O"),
     ("diagnosticado","O"),("com","O"),("hiperadrenocorticismo","B-DOENCA"),(".",  "O")],

    [("A","O"),("implantação","B-TRATAMENTO"),("de","I-TRATAMENTO"),
     ("deslorelina","B-MEDICAMENTO"),("foi","O"),("indicada","O"),
     ("para","O"),("o","O"),("furão","B-ESPECIE"),("com","O"),
     ("hiperadrenocorticismo","B-DOENCA"),(".",  "O")],

    # ── OFTALMOLOGIA ─────────────────────────────────────────────
    [("O","O"),("cão","B-ESPECIE"),("apresentou","O"),("opacidade","B-SINTOMA"),
     ("progressiva","I-SINTOMA"),("do","O"),("cristalino","O"),
     ("compatível","O"),("com","O"),("catarata","B-DOENCA"),("senil","I-DOENCA"),(".",  "O")],

    [("A","O"),("facoemulsificação","B-TRATAMENTO"),("foi","O"),("realizada","O"),
     ("para","O"),("remoção","O"),("da","O"),("catarata","B-DOENCA"),
     ("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("Cocker","B-ESPECIE"),("Spaniel","I-ESPECIE"),("apresentou","O"),
     ("glaucoma","B-DOENCA"),("primário","I-DOENCA"),("com","O"),
     ("pressão","B-SINTOMA"),("intraocular","I-SINTOMA"),("elevada","I-SINTOMA"),
     ("e","O"),("midríase","B-SINTOMA"),(".",  "O")],

    [("Dorzolamida","B-MEDICAMENTO"),("e","O"),("timolol","B-MEDICAMENTO"),
     ("foram","O"),("prescritos","O"),("para","O"),("controle","O"),
     ("do","O"),("glaucoma","B-DOENCA"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("com","O"),("uveíte","B-DOENCA"),
     ("anterior","I-DOENCA"),("apresentou","O"),("fotofobia","B-SINTOMA"),(",","O"),
     ("epífora","B-SINTOMA"),("e","O"),("miose","B-SINTOMA"),(".",  "O")],

    [("Prednisolona","B-MEDICAMENTO"),("tópica","O"),("foi","O"),("indicada","O"),
     ("para","O"),("controle","O"),("da","O"),("uveíte","B-DOENCA"),
     ("anterior","I-DOENCA"),("felina","O"),(".",  "O")],

    # ── ORTOPEDIA E EMERGÊNCIA ────────────────────────────────────
    [("O","O"),("filhote","B-ESPECIE"),("chegou","O"),("em","O"),("colapso","B-SINTOMA"),
     ("com","O"),("hipoglicemia","B-SINTOMA"),("grave","I-SINTOMA"),("e","O"),
     ("hipotermia","B-SINTOMA"),(".",  "O")],

    [("A","O"),("suplementação","B-TRATAMENTO"),("intravenosa","I-TRATAMENTO"),
     ("de","I-TRATAMENTO"),("glicose","B-MEDICAMENTO"),("foi","O"),("iniciada","O"),
     ("imediatamente","O"),("para","O"),("o","O"),("filhote","B-ESPECIE"),
     ("hipoglicêmico","O"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("foi","O"),("atendido","O"),("com","O"),
     ("trauma","B-DOENCA"),("cranioencefálico","I-DOENCA"),("após","O"),
     ("atropelamento","O"),(".",  "O")],

    [("O","O"),("mannitol","B-MEDICAMENTO"),("foi","O"),("administrado","O"),
     ("para","O"),("reduzir","O"),("o","O"),("edema","B-DOENCA"),
     ("cerebral","I-DOENCA"),("no","O"),("cão","B-ESPECIE"),
     ("politraumatizado","O"),(".",  "O")],

    [("O","O"),("Dachshund","B-ESPECIE"),("apresentou","O"),("paraplegia","B-SINTOMA"),
     ("aguda","I-SINTOMA"),("com","O"),("ausência","O"),("de","O"),
     ("nocicepção","O"),("profunda","O"),(".",  "O")],

    [("A","O"),("janela","O"),("terapêutica","O"),("para","O"),
     ("descompressão","B-TRATAMENTO"),("cirúrgica","I-TRATAMENTO"),
     ("é","O"),("de","O"),("até","O"),("48","O"),("horas","O"),
     ("na","O"),("DDIV","B-DOENCA"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("foi","O"),("trazido","O"),("com","O"),
     ("obstrução","B-DOENCA"),("uretral","I-DOENCA"),("e","O"),
     ("bexiga","B-SINTOMA"),("muito","O"),("distendida","I-SINTOMA"),(".",  "O")],

    [("O","O"),("cateterismo","B-TRATAMENTO"),("uretral","I-TRATAMENTO"),("foi","O"),
     ("realizado","O"),("para","O"),("desobstrução","O"),("no","O"),
     ("felino","B-ESPECIE"),(".",  "O")],

    # ── MEDICINA PREVENTIVA E NUTRIÇÃO ────────────────────────────
    [("A","O"),("vacinação","B-TRATAMENTO"),("antirrábica","I-TRATAMENTO"),("é","O"),
     ("obrigatória","O"),("para","O"),("cães","B-ESPECIE"),("e","O"),
     ("gatos","B-ESPECIE"),("em","O"),("todo","O"),("o","O"),("Brasil","O"),(".",  "O")],

    [("O","O"),("protocolo","O"),("vacinal","O"),("contra","O"),
     ("panleucopenia","B-DOENCA"),(",","O"),("herpesvírus","B-DOENCA"),("e","O"),
     ("calicivírus","B-DOENCA"),("é","O"),("essencial","O"),
     ("em","O"),("gatos","B-ESPECIE"),(".",  "O")],

    [("A","O"),("vermifugação","B-TRATAMENTO"),("regular","I-TRATAMENTO"),
     ("com","O"),("milbemicina","B-MEDICAMENTO"),("previne","O"),
     ("ancilostomíase","B-DOENCA"),("e","O"),("toxocaríase","B-DOENCA"),
     ("em","O"),("cães","B-ESPECIE"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("obeso","O"),("foi","O"),("submetido","O"),
     ("a","O"),("dieta","B-TRATAMENTO"),("hipocalórica","I-TRATAMENTO"),
     ("com","O"),("alto","O"),("teor","O"),("de","O"),("proteína","O"),(".",  "O")],

    [("A","O"),("deficiência","B-DOENCA"),("de","I-DOENCA"),("taurina","B-MEDICAMENTO"),
     ("em","O"),("gatos","B-ESPECIE"),("pode","O"),("causar","O"),
     ("cardiomiopatia","B-DOENCA"),("dilatada","I-DOENCA"),(".",  "O")],

    [("A","O"),("dieta","B-TRATAMENTO"),("úmida","I-TRATAMENTO"),("é","O"),
     ("recomendada","O"),("para","O"),("gatos","B-ESPECIE"),("com","O"),
     ("doença","B-DOENCA"),("renal","I-DOENCA"),("crônica","I-DOENCA"),(".",  "O")],

    [("O","O"),("escore","O"),("de","O"),("condição","O"),("corporal","O"),
     ("foi","O"),("avaliado","O"),("como","O"),("7","O"),("de","O"),("9","O"),
     ("no","O"),("cão","B-ESPECIE"),("com","O"),("obesidade","B-DOENCA"),(".",  "O")],

    # ── DERMATOLOGIA EXTRA ────────────────────────────────────────
    [("A","O"),("Malassezia","B-DOENCA"),("pachydermatis","I-DOENCA"),("foi","O"),
     ("identificada","O"),("em","O"),("citologia","O"),("do","O"),
     ("conduto","O"),("auditivo","O"),("do","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("Cetoconazol","B-MEDICAMENTO"),("shampoo","O"),("foi","O"),("prescrito","O"),
     ("para","O"),("dermatite","B-DOENCA"),("por","I-DOENCA"),
     ("Malassezia","I-DOENCA"),("no","O"),("Basset","B-ESPECIE"),("Hound","I-ESPECIE"),(".",  "O")],

    [("O","O"),("felino","B-ESPECIE"),("com","O"),("pênfigo","B-DOENCA"),
     ("foliáceo","I-DOENCA"),("apresentou","O"),("crostas","B-SINTOMA"),
     ("e","O"),("erosões","B-SINTOMA"),("na","O"),("face","O"),("e","O"),("orelhas","O"),(".",  "O")],

    [("Ciclosporina","B-MEDICAMENTO"),("e","O"),("dexametasona","B-MEDICAMENTO"),
     ("foram","O"),("usadas","O"),("para","O"),("o","O"),("pênfigo","B-DOENCA"),
     ("foliáceo","I-DOENCA"),("no","O"),("gato","B-ESPECIE"),(".",  "O")],

    [("O","O"),("Labrador","B-ESPECIE"),("apresentou","O"),("urticária","B-DOENCA"),
     ("aguda","I-DOENCA"),("com","O"),("edema","B-SINTOMA"),("de","I-SINTOMA"),
     ("face","I-SINTOMA"),("após","O"),("vacinação","O"),(".",  "O")],

    [("Adrenalina","B-MEDICAMENTO"),("e","O"),("difenidramina","B-MEDICAMENTO"),
     ("foram","O"),("administradas","O"),("na","O"),("reação","B-DOENCA"),
     ("anafilática","I-DOENCA"),("do","O"),("cão","B-ESPECIE"),(".",  "O")],

    # ── REPRODUÇÃO ───────────────────────────────────────────────
    [("A","O"),("cadela","B-ESPECIE"),("apresentou","O"),("descarga","B-SINTOMA"),
     ("vaginal","I-SINTOMA"),("purulenta","I-SINTOMA"),("e","O"),("poliúria","B-SINTOMA"),
     ("compatíveis","O"),("com","O"),("piómetra","B-DOENCA"),("aberta","I-DOENCA"),(".",  "O")],

    [("A","O"),("ovário-histerectomia","B-TRATAMENTO"),("de","I-TRATAMENTO"),
     ("emergência","I-TRATAMENTO"),("foi","O"),("indicada","O"),("para","O"),
     ("a","O"),("cadela","B-ESPECIE"),("com","O"),("piómetra","B-DOENCA"),(".",  "O")],

    [("O","O"),("criptorquidismo","B-DOENCA"),("bilateral","I-DOENCA"),("foi","O"),
     ("diagnosticado","O"),("no","O"),("cão","B-ESPECIE"),
     ("durante","O"),("exame","O"),("pré-operatório","O"),(".",  "O")],

    [("A","O"),("orquiectomia","B-TRATAMENTO"),("abdominal","I-TRATAMENTO"),
     ("foi","O"),("realizada","O"),("para","O"),("remoção","O"),
     ("dos","O"),("testículos","O"),("retidos","O"),
     ("no","O"),("cão","B-ESPECIE"),("com","O"),("criptorquidismo","B-DOENCA"),(".",  "O")],

    # ── GASTROENTEROLOGIA ────────────────────────────────────────
    [("O","O"),("cão","B-ESPECIE"),("apresentou","O"),("diarreia","B-SINTOMA"),
     ("crônica","I-SINTOMA"),(",","O"),("perda","B-SINTOMA"),("de","I-SINTOMA"),
     ("peso","I-SINTOMA"),("e","O"),("hipoalbuminemia","B-SINTOMA"),
     ("compatíveis","O"),("com","O"),("enteropatia","B-DOENCA"),
     ("perdedora","I-DOENCA"),("de","I-DOENCA"),("proteína","I-DOENCA"),(".",  "O")],

    [("A","O"),("endoscopia","B-TRATAMENTO"),("com","O"),("biópsia","B-TRATAMENTO"),
     ("intestinal","I-TRATAMENTO"),("confirmou","O"),("a","O"),
     ("doença","B-DOENCA"),("inflamatória","I-DOENCA"),("intestinal","I-DOENCA"),
     ("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("com","O"),("pancreatite","B-DOENCA"),
     ("aguda","I-DOENCA"),("apresentou","O"),("vômito","B-SINTOMA"),(",","O"),
     ("dor","B-SINTOMA"),("abdominal","I-SINTOMA"),("e","O"),
     ("anorexia","B-SINTOMA"),("intensa","I-SINTOMA"),(".",  "O")],

    [("Maropitant","B-MEDICAMENTO"),("e","O"),("ondansetrona","B-MEDICAMENTO"),
     ("foram","O"),("usados","O"),("para","O"),("controle","O"),
     ("dos","O"),("vômitos","B-SINTOMA"),("na","O"),("pancreatite","B-DOENCA"),
     ("felina","O"),(".",  "O")],

    [("O","O"),("Boxer","B-ESPECIE"),("apresentou","O"),("megaesôfago","B-DOENCA"),
     ("com","O"),("regurgitação","B-SINTOMA"),("pós-prandial","I-SINTOMA"),
     ("e","O"),("pneumonia","B-DOENCA"),("por","I-DOENCA"),
     ("aspiração","I-DOENCA"),("recorrente","O"),(".",  "O")],

    [("A","O"),("alimentação","B-TRATAMENTO"),("em","I-TRATAMENTO"),
     ("posição","I-TRATAMENTO"),("vertical","I-TRATAMENTO"),("foi","O"),
     ("recomendada","O"),("para","O"),("o","O"),("cão","B-ESPECIE"),
     ("com","O"),("megaesôfago","B-DOENCA"),(".",  "O")],

    # ── PNEUMOLOGIA ──────────────────────────────────────────────
    [("O","O"),("cão","B-ESPECIE"),("com","O"),("colapso","B-DOENCA"),
     ("de","I-DOENCA"),("traqueia","I-DOENCA"),("apresentou","O"),
     ("tosse","B-SINTOMA"),("em","O"),("ganso","O"),("e","O"),
     ("cianose","B-SINTOMA"),("ao","O"),("exercício","O"),(".",  "O")],

    [("O","O"),("stent","B-TRATAMENTO"),("traqueal","I-TRATAMENTO"),
     ("foi","O"),("implantado","O"),("para","O"),("o","O"),
     ("colapso","B-DOENCA"),("de","I-DOENCA"),("traqueia","I-DOENCA"),
     ("no","O"),("Yorkshire","B-ESPECIE"),(".",  "O")],

    [("A","O"),("efusão","B-DOENCA"),("pleural","I-DOENCA"),("causou","O"),
     ("dispneia","B-SINTOMA"),("e","O"),("respiração","B-SINTOMA"),
     ("abdominal","I-SINTOMA"),("no","O"),("gato","B-ESPECIE"),(".",  "O")],

    [("A","O"),("toracocentese","B-TRATAMENTO"),("bilateral","I-TRATAMENTO"),
     ("removeu","O"),("800","O"),("ml","O"),("de","O"),("líquido","O"),
     ("do","O"),("tórax","O"),("do","O"),("felino","B-ESPECIE"),(".",  "O")],

    # ── TOXICOLOGIA ──────────────────────────────────────────────
    [("O","O"),("cão","B-ESPECIE"),("ingeriu","O"),("rodenticida","B-DOENCA"),
     ("e","O"),("apresentou","O"),("sangramento","B-SINTOMA"),("espontâneo","I-SINTOMA"),
     ("e","O"),("hematomas","B-SINTOMA"),(".",  "O")],

    [("Vitamina","B-MEDICAMENTO"),("K1","I-MEDICAMENTO"),("foi","O"),("administrada","O"),
     ("para","O"),("reverter","O"),("a","O"),("coagulopatia","B-DOENCA"),
     ("por","I-DOENCA"),("rodenticida","I-DOENCA"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("apresentou","O"),("insuficiência","B-DOENCA"),
     ("renal","I-DOENCA"),("aguda","I-DOENCA"),("após","O"),("ingestão","O"),
     ("de","O"),("lírios","O"),(".",  "O")],

    [("A","O"),("diurese","B-TRATAMENTO"),("forçada","I-TRATAMENTO"),("com","O"),
     ("fluidoterapia","B-MEDICAMENTO"),("intensiva","O"),("foi","O"),
     ("iniciada","O"),("no","O"),("gato","B-ESPECIE"),("intoxicado","O"),
     ("por","O"),("lírios","O"),(".",  "O")],

    [("A","O"),("reabilitação","B-TRATAMENTO"),("física","I-TRATAMENTO"),
     ("foi","O"),("indicada","O"),("após","O"),("a","O"),("cirurgia","O"),
     ("de","O"),("ligamento","O"),("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("recebeu","O"),("acupuntura","B-TRATAMENTO"),
     ("para","O"),("alívio","O"),("de","O"),("dor","B-SINTOMA"),("crônica","I-SINTOMA"),
     ("nas","O"),("articulações","O"),(".",  "O")],

    [("A","O"),("profilaxia","B-TRATAMENTO"),("antibiótica","I-TRATAMENTO"),
     ("foi","O"),("administrada","O"),("antes","O"),("da","O"),
     ("cirurgia","O"),("cardíaca","O"),("no","O"),("gato","B-ESPECIE"),(".",  "O")],

    [("O","O"),("gato","B-ESPECIE"),("foi","O"),("submetido","O"),
     ("à","O"),("esterilização","B-TRATAMENTO"),("precoce","I-TRATAMENTO"),
     ("aos","O"),("três","O"),("meses","O"),("de","O"),("idade","O"),(".",  "O")],

    [("A","O"),("drenagem","B-TRATAMENTO"),("de","I-TRATAMENTO"),
     ("abscessos","I-TRATAMENTO"),("foi","O"),("realizada","O"),
     ("no","O"),("cão","B-ESPECIE"),("com","O"),("piodermite","B-DOENCA"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("com","O"),("epilepsia","B-DOENCA"),
     ("foi","O"),("submetido","O"),("a","O"),("estimulação","B-TRATAMENTO"),
     ("neural","I-TRATAMENTO"),("vagal","I-TRATAMENTO"),("(",  "O"),("VNS","O"),
     (")",  "O"),(".",  "O")],

    [("A","O"),("cauterização","B-TRATAMENTO"),("endoscópica","I-TRATAMENTO"),
     ("removeu","O"),("o","O"),("hemangioma","B-DOENCA"),("do","O"),
     ("duodeno","O"),("no","O"),("gato","B-ESPECIE"),(".",  "O")],

    [("Suportive","O"),("care","O"),("com","O"),("fluidos","B-TRATAMENTO"),
     ("e","O"),("nutrição","B-TRATAMENTO"),("foi","O"),
     ("essencial","O"),("durante","O"),("o","O"),("tratamento","O"),
     ("da","O"),("parvovirose","B-DOENCA"),(".",  "O")],

    [("A","O"),("ultrassonografia","B-TRATAMENTO"),("terapêutica","I-TRATAMENTO"),
     ("foi","O"),("usada","O"),("para","O"),("destruição","O"),
     ("de","O"),("tumores","B-DOENCA"),("de","I-DOENCA"),("pele","I-DOENCA"),
     ("no","O"),("cão","B-ESPECIE"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("apresentava","O"),("artrose","B-DOENCA"),
     ("avançada","I-DOENCA"),("e","O"),("recebeu","O"),
     ("infiltração","B-TRATAMENTO"),("articular","I-TRATAMENTO"),
     ("de","I-TRATAMENTO"),("ácido","B-MEDICAMENTO"),("hialurônico","I-MEDICAMENTO"),(".",  "O")],

    [("A","O"),("amputação","B-TRATAMENTO"),("de","I-TRATAMENTO"),
     ("cauda","I-TRATAMENTO"),("foi","O"),("necessária","O"),
     ("no","O"),("gato","B-ESPECIE"),("com","O"),("trauma","B-DOENCA"),
     ("severo","I-DOENCA"),(".",  "O")],

    [("A","O"),("plastyplastia","B-TRATAMENTO"),("de","I-TRATAMENTO"),
     ("pálato","I-TRATAMENTO"),("foi","O"),("realizada","O"),
     ("no","O"),("bulldog","B-ESPECIE"),("com","O"),
     ("síndrome","B-DOENCA"),("braquicéfala","I-DOENCA"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("recebeu","O"),("fototerapia","B-TRATAMENTO"),
     ("com","O"),("laser","O"),("para","O"),("cicatrização","O"),
     ("de","O"),("ferida","B-DOENCA"),("traumática","I-DOENCA"),(".",  "O")],

    [("A","O"),("castração","B-TRATAMENTO"),("química","I-TRATAMENTO"),
     ("com","O"),("deslorelina","B-MEDICAMENTO"),("foi","O"),("uma","O"),
     ("alternativa","O"),("considerada","O"),("no","O"),("gato","B-ESPECIE"),(".",  "O")],

    [("A","O"),("antiotoxina","B-TRATAMENTO"),("foi","O"),("administrada","O"),
     ("imediatamente","O"),("no","O"),("cão","B-ESPECIE"),("com","O"),
     ("intoxicação","B-DOENCA"),("tetânica","I-DOENCA"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("com","O"),("sarna","B-DOENCA"),
     ("demodécica","I-DOENCA"),("foi","O"),("submetido","O"),
     ("a","O"),("múltiplos","O"),("banhos","B-TRATAMENTO"),
     ("acaricidas","I-TRATAMENTO"),("semanais","I-TRATAMENTO"),(".",  "O")],

    [("A","O"),("ressonância","B-TRATAMENTO"),("magnética","I-TRATAMENTO"),
     ("revelou","O"),("uma","O"),("hérnia","B-DOENCA"),("discal","I-DOENCA"),
     ("no","O"),("gato","B-ESPECIE"),(".",  "O")],

    [("A","O"),("pneumopeonia","B-TRATAMENTO"),("foi","O"),("utilizada","O"),
     ("como","O"),("técnica","O"),("diagnóstica","O"),("em","O"),
     ("suspeita","O"),("de","O"),("peritonite","B-DOENCA"),("infecciosa","I-DOENCA"),(".",  "O")],

    [("O","O"),("cão","B-ESPECIE"),("idoso","O"),("com","O"),
     ("insuficiência","B-DOENCA"),("cardíaca","I-DOENCA"),("recebeu","O"),
     ("oxigenoterapia","B-TRATAMENTO"),("contínua","I-TRATAMENTO"),
     ("durante","O"),("hospitalização","O"),(".",  "O")],

    [("A","O"),("flebotomia","B-TRATAMENTO"),("terapêutica","I-TRATAMENTO"),
     ("removeu","O"),("sangue","O"),("para","O"),("reduzir","O"),
     ("a","O"),("poliglobulia","B-DOENCA"),("no","O"),("gato","B-ESPECIE"),(".",  "O")],
]


class VetNERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = zip(*self.data[idx])
        tokens = list(tokens)
        tags   = list(tags)

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word:
                aligned_labels.append(LABEL2ID[tags[word_id]])
            else:
                label = tags[word_id]
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                aligned_labels.append(LABEL2ID.get(label, -100))
            prev_word = word_id

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
        }


def get_splits(data=RAW_DATA, train_ratio=0.85, seed=42):
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def save_dataset(path="data/dataset.json"):
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(RAW_DATA, f, ensure_ascii=False, indent=2)
    print(f"Dataset salvo em {path} ({len(RAW_DATA)} sentenças)")


if __name__ == "__main__":
    save_dataset()
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    train, val = get_splits()
    print(f"Treino: {len(train)} | Validação: {len(val)}")
    ds = VetNERDataset(train, tokenizer)
    sample = ds[0]
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"Labels : {[ID2LABEL[l.item()] for l in sample['labels'] if l.item() != -100]}")
