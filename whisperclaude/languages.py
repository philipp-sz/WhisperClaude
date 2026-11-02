"""Languages for the translate key (NLLB-200 codes) and the picker's search. Light: no model.

Names and native names were generated from the CLDR data (langcodes) and cleaned by hand.
"""
from __future__ import annotations

import unicodedata
from typing import NamedTuple


class Language(NamedTuple):
    code: str    # NLLB code, e.g. "deu_Latn"
    name: str    # English name
    native: str  # name in the language itself ("" if the same as the English one)


NLLB_LANGUAGES: tuple[tuple[str, str, str], ...] = (
    ('ace_Arab', 'Acehnese (Arabic)', ''),
    ('ace_Latn', 'Acehnese (Latin)', ''),
    ('afr_Latn', 'Afrikaans', ''),
    ('aka_Latn', 'Akan', ''),
    ('als_Latn', 'Albanian', 'shqip'),
    ('amh_Ethi', 'Amharic', 'አማርኛ'),
    ('arb_Arab', 'Arabic', 'العربية'),
    ('hye_Armn', 'Armenian', 'հայերեն'),
    ('asm_Beng', 'Assamese', 'অসমীয়া'),
    ('ast_Latn', 'Asturian', 'asturianu'),
    ('awa_Deva', 'Awadhi', ''),
    ('ayr_Latn', 'Aymara', ''),
    ('azj_Latn', 'Azerbaijani', 'azərbaycan'),
    ('ban_Latn', 'Balinese', ''),
    ('bam_Latn', 'Bambara', 'bamanakan'),
    ('bjn_Arab', 'Banjar (Arabic)', ''),
    ('bjn_Latn', 'Banjar (Latin)', ''),
    ('bak_Cyrl', 'Bashkir', 'башҡорт теле'),
    ('eus_Latn', 'Basque', 'euskara'),
    ('bel_Cyrl', 'Belarusian', 'беларуская'),
    ('bem_Latn', 'Bemba', 'Ichibemba'),
    ('ben_Beng', 'Bengali', 'বাংলা'),
    ('bho_Deva', 'Bhojpuri', 'भोजपुरी'),
    ('bos_Latn', 'Bosnian', 'bosanski'),
    ('bug_Latn', 'Buginese', ''),
    ('bul_Cyrl', 'Bulgarian', 'български'),
    ('mya_Mymr', 'Burmese', 'မြန်မာ'),
    ('yue_Hant', 'Cantonese', '粵語'),
    ('cat_Latn', 'Catalan', 'català'),
    ('ceb_Latn', 'Cebuano', ''),
    ('tzm_Tfng', 'Central Atlas Tamazight', 'Tamaziɣt n laṭlaṣ'),
    ('hne_Deva', 'Chhattisgarhi', ''),
    ('nya_Latn', 'Chichewa (Nyanja)', 'Nyanja'),
    ('zho_Hans', 'Chinese (Simplified)', '中文'),
    ('zho_Hant', 'Chinese (Traditional)', '中文'),
    ('cjk_Latn', 'Chokwe', ''),
    ('crh_Latn', 'Crimean Tatar', ''),
    ('hrv_Latn', 'Croatian', 'hrvatski'),
    ('ces_Latn', 'Czech', 'čeština'),
    ('dan_Latn', 'Danish', 'dansk'),
    ('prs_Arab', 'Dari', 'دری'),
    ('dik_Latn', 'Dinka', ''),
    ('nld_Latn', 'Dutch', 'Nederlands'),
    ('dyu_Latn', 'Dyula', ''),
    ('dzo_Tibt', 'Dzongkha', 'རྫོང་ཁ'),
    ('arz_Arab', 'Egyptian Arabic', ''),
    ('eng_Latn', 'English', ''),
    ('epo_Latn', 'Esperanto', ''),
    ('est_Latn', 'Estonian', 'eesti'),
    ('ewe_Latn', 'Ewe', 'eʋegbe'),
    ('fao_Latn', 'Faroese', 'føroyskt'),
    ('fij_Latn', 'Fijian', ''),
    ('tgl_Latn', 'Filipino (Tagalog)', ''),
    ('fin_Latn', 'Finnish', 'suomi'),
    ('fon_Latn', 'Fon', ''),
    ('fra_Latn', 'French', 'français'),
    ('fur_Latn', 'Friulian', 'furlan'),
    ('glg_Latn', 'Galician', 'galego'),
    ('lug_Latn', 'Ganda', 'Luganda'),
    ('kat_Geor', 'Georgian', 'ქართული'),
    ('deu_Latn', 'German', 'Deutsch'),
    ('ell_Grek', 'Greek', 'Ελληνικά'),
    ('grn_Latn', 'Guarani', 'avañe’ẽ'),
    ('guj_Gujr', 'Gujarati', 'ગુજરાતી'),
    ('hat_Latn', 'Haitian Creole', 'Kreyòl Ayisyen'),
    ('hau_Latn', 'Hausa', ''),
    ('heb_Hebr', 'Hebrew', 'עברית'),
    ('hin_Deva', 'Hindi', 'हिन्दी'),
    ('hun_Latn', 'Hungarian', 'magyar'),
    ('isl_Latn', 'Icelandic', 'íslenska'),
    ('ibo_Latn', 'Igbo', ''),
    ('ilo_Latn', 'Ilocano', 'Ilokano'),
    ('ind_Latn', 'Indonesian', 'bahasa Indonesia'),
    ('gle_Latn', 'Irish', 'Gaeilge'),
    ('ita_Latn', 'Italian', 'italiano'),
    ('jpn_Jpan', 'Japanese', '日本語'),
    ('jav_Latn', 'Javanese', 'Jawa'),
    ('kbp_Latn', 'Kabiyè', ''),
    ('kea_Latn', 'Kabuverdianu', ''),
    ('kab_Latn', 'Kabyle', 'Taqbaylit'),
    ('kac_Latn', 'Kachin', ''),
    ('kam_Latn', 'Kamba', 'Kikamba'),
    ('kan_Knda', 'Kannada', 'ಕನ್ನಡ'),
    ('knc_Arab', 'Kanuri (Arabic)', ''),
    ('knc_Latn', 'Kanuri (Latin)', ''),
    ('kas_Arab', 'Kashmiri (Arabic)', 'کٲشُر'),
    ('kas_Deva', 'Kashmiri (Devanagari)', 'کٲشُر'),
    ('kaz_Cyrl', 'Kazakh', 'қазақ тілі'),
    ('khm_Khmr', 'Khmer', 'ខ្មែរ'),
    ('kik_Latn', 'Kikuyu', 'Gikuyu'),
    ('kmb_Latn', 'Kimbundu', ''),
    ('kin_Latn', 'Kinyarwanda', 'Ikinyarwanda'),
    ('kon_Latn', 'Kongo', ''),
    ('kor_Hang', 'Korean', '한국어'),
    ('kmr_Latn', 'Kurdish (Kurmanji)', 'kurdî (kurmancî)'),
    ('ckb_Arab', 'Kurdish (Sorani)', 'کوردیی ناوەندی'),
    ('kir_Cyrl', 'Kyrgyz', 'кыргызча'),
    ('lao_Laoo', 'Lao', 'ລາວ'),
    ('ltg_Latn', 'Latgalian', ''),
    ('lvs_Latn', 'Latvian', 'latviešu'),
    ('lij_Latn', 'Ligurian', 'ligure'),
    ('lim_Latn', 'Limburgish', ''),
    ('lin_Latn', 'Lingala', 'lingála'),
    ('lit_Latn', 'Lithuanian', 'lietuvių'),
    ('lmo_Latn', 'Lombard', ''),
    ('lua_Latn', 'Luba-Lulua', ''),
    ('luo_Latn', 'Luo', 'Dholuo'),
    ('ltz_Latn', 'Luxembourgish', 'Lëtzebuergesch'),
    ('mkd_Cyrl', 'Macedonian', 'македонски'),
    ('mag_Deva', 'Magahi', ''),
    ('mai_Deva', 'Maithili', 'मैथिली'),
    ('plt_Latn', 'Malagasy', ''),
    ('zsm_Latn', 'Malay', 'bahasa Malaysia'),
    ('mal_Mlym', 'Malayalam', 'മലയാളം'),
    ('mlt_Latn', 'Maltese', 'Malti'),
    ('mni_Beng', 'Manipuri', 'মৈতৈলোন্'),
    ('mri_Latn', 'Maori', 'Māori'),
    ('mar_Deva', 'Marathi', 'मराठी'),
    ('acm_Arab', 'Mesopotamian Arabic', ''),
    ('min_Latn', 'Minangkabau', ''),
    ('lus_Latn', 'Mizo', ''),
    ('khk_Cyrl', 'Mongolian', 'монгол'),
    ('ary_Arab', 'Moroccan Arabic', ''),
    ('mos_Latn', 'Mossi', ''),
    ('ars_Arab', 'Najdi Arabic', ''),
    ('npi_Deva', 'Nepali', 'नेपाली'),
    ('fuv_Latn', 'Nigerian Fulfulde', ''),
    ('apc_Arab', 'North Levantine Arabic', 'العامية'),
    ('nso_Latn', 'Northern Sotho', 'Sesotho sa Leboa'),
    ('nob_Latn', 'Norwegian (Bokmål)', 'norsk bokmål'),
    ('nno_Latn', 'Norwegian (Nynorsk)', 'norsk nynorsk'),
    ('nus_Latn', 'Nuer', 'Thok Nath'),
    ('oci_Latn', 'Occitan', ''),
    ('ory_Orya', 'Odia', 'ଓଡ଼ିଆ'),
    ('gaz_Latn', 'Oromo', 'Oromoo'),
    ('pag_Latn', 'Pangasinan', ''),
    ('pap_Latn', 'Papiamento', 'Papiamentu'),
    ('pbt_Arab', 'Pashto', 'Southern Pashto'),
    ('pes_Arab', 'Persian', 'فارسی'),
    ('pol_Latn', 'Polish', 'polski'),
    ('por_Latn', 'Portuguese', 'português'),
    ('pan_Guru', 'Punjabi', 'ਪੰਜਾਬੀ'),
    ('quy_Latn', 'Quechua', 'Ayacucho Quechua'),
    ('ron_Latn', 'Romanian', 'română'),
    ('run_Latn', 'Rundi', 'Ikirundi'),
    ('rus_Cyrl', 'Russian', 'русский'),
    ('smo_Latn', 'Samoan', ''),
    ('sag_Latn', 'Sango', 'Sängö'),
    ('san_Deva', 'Sanskrit', 'संस्कृत भाषा'),
    ('srd_Latn', 'Sardinian', 'sardu'),
    ('gla_Latn', 'Scottish Gaelic', 'Gàidhlig'),
    ('srp_Cyrl', 'Serbian', 'српски'),
    ('shn_Mymr', 'Shan', 'တႆး'),
    ('sna_Latn', 'Shona', 'chiShona'),
    ('scn_Latn', 'Sicilian', 'sicilianu'),
    ('szl_Latn', 'Silesian', 'ślōnski'),
    ('snd_Arab', 'Sindhi', 'سنڌي'),
    ('sin_Sinh', 'Sinhala', 'සිංහල'),
    ('slk_Latn', 'Slovak', 'slovenčina'),
    ('slv_Latn', 'Slovenian', 'slovenščina'),
    ('som_Latn', 'Somali', 'Soomaali'),
    ('azb_Arab', 'South Azerbaijani', ''),
    ('ajp_Arab', 'South Levantine Arabic', ''),
    ('sot_Latn', 'Southern Sotho', 'Sesotho'),
    ('spa_Latn', 'Spanish', 'español'),
    ('sun_Latn', 'Sundanese', 'Basa Sunda'),
    ('swh_Latn', 'Swahili', 'Kiswahili'),
    ('ssw_Latn', 'Swati', 'siSwati'),
    ('swe_Latn', 'Swedish', 'svenska'),
    ('acq_Arab', "Ta'izzi-Adeni Arabic", ''),
    ('tgk_Cyrl', 'Tajik', 'тоҷикӣ'),
    ('taq_Latn', 'Tamasheq (Latin)', ''),
    ('taq_Tfng', 'Tamasheq (Tifinagh)', ''),
    ('tam_Taml', 'Tamil', 'தமிழ்'),
    ('tat_Cyrl', 'Tatar', 'татар'),
    ('tel_Telu', 'Telugu', 'తెలుగు'),
    ('tha_Thai', 'Thai', 'ไทย'),
    ('bod_Tibt', 'Tibetan', 'བོད་སྐད་'),
    ('tir_Ethi', 'Tigrinya', 'ትግርኛ'),
    ('tpi_Latn', 'Tok Pisin', ''),
    ('tso_Latn', 'Tsonga', ''),
    ('tsn_Latn', 'Tswana', 'Setswana'),
    ('tum_Latn', 'Tumbuka', ''),
    ('aeb_Arab', 'Tunisian Arabic', ''),
    ('tur_Latn', 'Turkish', 'Türkçe'),
    ('tuk_Latn', 'Turkmen', 'türkmen dili'),
    ('twi_Latn', 'Twi', 'Akan'),
    ('ukr_Cyrl', 'Ukrainian', 'українська'),
    ('umb_Latn', 'Umbundu', ''),
    ('urd_Arab', 'Urdu', 'اردو'),
    ('uig_Arab', 'Uyghur', 'ئۇيغۇرچە'),
    ('uzn_Latn', 'Uzbek', 'o‘zbek'),
    ('vec_Latn', 'Venetian', 'veneto'),
    ('vie_Latn', 'Vietnamese', 'Tiếng Việt'),
    ('war_Latn', 'Waray', ''),
    ('cym_Latn', 'Welsh', 'Cymraeg'),
    ('wol_Latn', 'Wolof', ''),
    ('xho_Latn', 'Xhosa', 'IsiXhosa'),
    ('ydd_Hebr', 'Yiddish', 'ייִדיש'),
    ('yor_Latn', 'Yoruba', 'Èdè Yorùbá'),
    ('zul_Latn', 'Zulu', 'isiZulu'),
)

# ISO 639 code (py3langid, Whisper) -> NLLB code
SOURCE_CODES = {
    'ace': 'ace_Latn', 'af': 'afr_Latn', 'ak': 'aka_Latn', 'am': 'amh_Ethi', 'ar': 'arb_Arab',
    'ary': 'ary_Arab', 'arz': 'arz_Arab', 'as': 'asm_Beng', 'ay': 'ayr_Latn', 'az': 'azj_Latn',
    'ba': 'bak_Cyrl', 'be': 'bel_Cyrl', 'bg': 'bul_Cyrl', 'bjn': 'bjn_Latn', 'bm': 'bam_Latn',
    'bn': 'ben_Beng', 'bo': 'bod_Tibt', 'bs': 'bos_Latn', 'ca': 'cat_Latn', 'crh': 'crh_Latn',
    'cs': 'ces_Latn', 'cy': 'cym_Latn', 'da': 'dan_Latn', 'de': 'deu_Latn', 'dz': 'dzo_Tibt',
    'ee': 'ewe_Latn', 'el': 'ell_Grek', 'en': 'eng_Latn', 'eo': 'epo_Latn', 'es': 'spa_Latn',
    'et': 'est_Latn', 'eu': 'eus_Latn', 'fa': 'pes_Arab', 'ff': 'fuv_Latn', 'fi': 'fin_Latn',
    'fj': 'fij_Latn', 'fo': 'fao_Latn', 'fr': 'fra_Latn', 'fuv': 'fuv_Latn', 'ga': 'gle_Latn',
    'gd': 'gla_Latn', 'gl': 'glg_Latn', 'gn': 'grn_Latn', 'gu': 'guj_Gujr', 'gug': 'grn_Latn',
    'ha': 'hau_Latn', 'he': 'heb_Hebr', 'hi': 'hin_Deva', 'hr': 'hrv_Latn', 'ht': 'hat_Latn',
    'hu': 'hun_Latn', 'hy': 'hye_Armn', 'id': 'ind_Latn', 'ig': 'ibo_Latn', 'is': 'isl_Latn',
    'it': 'ita_Latn', 'ja': 'jpn_Jpan', 'jv': 'jav_Latn', 'ka': 'kat_Geor', 'kab': 'kab_Latn',
    'kg': 'kon_Latn', 'ki': 'kik_Latn', 'kk': 'kaz_Cyrl', 'km': 'khm_Khmr', 'kn': 'kan_Knda',
    'ko': 'kor_Hang', 'kr': 'knc_Latn', 'ks': 'kas_Arab', 'ku': 'kmr_Latn', 'ky': 'kir_Cyrl',
    'lb': 'ltz_Latn', 'lg': 'lug_Latn', 'li': 'lim_Latn', 'lij': 'lij_Latn', 'ln': 'lin_Latn',
    'lo': 'lao_Laoo', 'lt': 'lit_Latn', 'ltg': 'ltg_Latn', 'lv': 'lvs_Latn', 'mg': 'plt_Latn',
    'mi': 'mri_Latn', 'min': 'min_Latn', 'mk': 'mkd_Cyrl', 'ml': 'mal_Mlym', 'mn': 'khk_Cyrl',
    'mr': 'mar_Deva', 'ms': 'zsm_Latn', 'mt': 'mlt_Latn', 'my': 'mya_Mymr', 'nb': 'nob_Latn',
    'ne': 'npi_Deva', 'nl': 'nld_Latn', 'nn': 'nno_Latn', 'no': 'nob_Latn', 'nso': 'nso_Latn',
    'ny': 'nya_Latn', 'oc': 'oci_Latn', 'om': 'gaz_Latn', 'or': 'ory_Orya', 'pa': 'pan_Guru',
    'pl': 'pol_Latn', 'ps': 'pbt_Arab', 'pt': 'por_Latn', 'qu': 'quy_Latn', 'rn': 'run_Latn',
    'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'rw': 'kin_Latn', 'sa': 'san_Deva', 'sc': 'srd_Latn',
    'sd': 'snd_Arab', 'sg': 'sag_Latn', 'si': 'sin_Sinh', 'sk': 'slk_Latn', 'sl': 'slv_Latn',
    'sm': 'smo_Latn', 'sn': 'sna_Latn', 'so': 'som_Latn', 'sq': 'als_Latn', 'sr': 'srp_Cyrl',
    'ss': 'ssw_Latn', 'st': 'sot_Latn', 'su': 'sun_Latn', 'sv': 'swe_Latn', 'sw': 'swh_Latn',
    'ta': 'tam_Taml', 'taq': 'taq_Latn', 'te': 'tel_Telu', 'tg': 'tgk_Cyrl', 'th': 'tha_Thai',
    'ti': 'tir_Ethi', 'tk': 'tuk_Latn', 'tl': 'tgl_Latn', 'tn': 'tsn_Latn', 'tr': 'tur_Latn',
    'ts': 'tso_Latn', 'tt': 'tat_Cyrl', 'tw': 'twi_Latn', 'ug': 'uig_Arab', 'uk': 'ukr_Cyrl',
    'ur': 'urd_Arab', 'uz': 'uzn_Latn', 'vec': 'vec_Latn', 'vi': 'vie_Latn', 'wo': 'wol_Latn',
    'xh': 'xho_Latn', 'yi': 'ydd_Hebr', 'yo': 'yor_Latn', 'yue': 'yue_Hant', 'zh': 'zho_Hans',
    'zu': 'zul_Latn',
}


NLLB_CODES = frozenset(code for code, _, _ in NLLB_LANGUAGES)
LANGUAGES = [Language(*row) for row in NLLB_LANGUAGES]
BY_CODE = {lang.code: lang for lang in LANGUAGES}
# Shown first (after the recent ones) and preferred on ties: widely used languages
COMMON = ("eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "nld_Latn",
          "pol_Latn", "ces_Latn", "rus_Cyrl", "ukr_Cyrl", "tur_Latn", "arb_Arab", "zho_Hans",
          "jpn_Jpan", "kor_Hang", "hin_Deva", "swe_Latn", "nob_Latn", "dan_Latn", "fin_Latn",
          "ell_Grek", "hun_Latn", "ron_Latn", "bul_Cyrl", "hrv_Latn", "srp_Cyrl", "slk_Latn",
          "slv_Latn", "lit_Latn", "lvs_Latn", "est_Latn", "heb_Hebr", "pes_Arab", "ind_Latn",
          "vie_Latn", "tha_Thai", "zho_Hant")
_COMMON = {code: i for i, code in enumerate(COMMON)}
# Typed in the search box: the ISO code finds the language too ("de" -> German)
_ISO = {}
for _iso, _code in SOURCE_CODES.items():
    _ISO.setdefault(_code, _iso)


def _fold(text: str) -> str:
    """Lower case without accents, so "francais" finds "français"."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def search(languages: list[Language], query: str, recent: list[str] = ()) -> list[Language]:
    """Languages matching the query, best first: exact code, name or native name starting
    with it, a word of it starting with it, anything containing it. Recently used languages
    come first within each group, then common ones; with an empty query: recent ones, then
    COMMON, then the rest by name."""
    rank = {code: i for i, code in enumerate(recent)}
    q = _fold(query.strip())

    def score(lang: Language) -> tuple | None:
        recency = rank.get(lang.code, len(rank))
        common = _COMMON.get(lang.code, len(_COMMON))
        if not q:
            return (recency, common, lang.name.casefold())
        names = [_fold(lang.name), _fold(lang.native)]
        codes = [lang.code.casefold(), _ISO.get(lang.code, "")]
        if q in codes:
            group = 0
        elif any(n.startswith(q) for n in names if n):
            group = 1
        elif any(w.startswith(q) for n in names for w in n.replace("(", " ").split()):
            group = 2
        elif any(q in n for n in names if n) or any(c.startswith(q) for c in codes if c):
            group = 3
        else:
            return None
        return (group, recency, common, lang.name.casefold())

    scored = [(s, lang) for lang in languages if (s := score(lang)) is not None]
    return [lang for _, lang in sorted(scored)]
