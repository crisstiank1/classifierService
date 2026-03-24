from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
import spacy
import unicodedata
import re
import os
import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classifier")

DATABASE_URL   = os.getenv("DATABASE_URL")

# ─── Carga modelo SpaCy ───────────────────────────────────────────────────────
try:
    nlp = spacy.load("es_core_news_sm")
    logger.info("✅ SpaCy model cargado correctamente")
except OSError:
    logger.warning("⚠️  SpaCy model no encontrado, usando keywords puro")
    nlp = None

app = FastAPI(title="DocuCloud Classifier", version="2.0")

app = FastAPI(title="DocuCloud Classifier", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── DTOs ─────────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    file_name: str
    mime_type: Optional[str] = None
    preview_text: Optional[str] = None
    size_bytes: Optional[int] = None


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    suggested_tags: List[str] = []
    rules_matched: List[str] = []


# ─── MEJORA 5: Pesos dinámicos por extensión ──────────────────────────────────
# 0.60 = casi diagnóstico  (extensión muy exclusiva de esa categoría)
# 0.35 = señal fuerte
# 0.20 = señal débil       (extensión compartida por muchas categorías)

EXTENSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Facturas": {".xml": 0.60, ".pdf": 0.25, ".xlsx": 0.20, ".docx": 0.20}, 
    "Contratos":  {".docx": 0.30, ".pdf": 0.20},
    "Informes":   {".pbix": 0.60, ".twbx": 0.60, ".csv": 0.35,
                   ".xlsx": 0.25, ".pptx": 0.20, ".pdf": 0.15},
    "Personal":   {".tiff": 0.50, ".tif": 0.50, ".heic": 0.35,
                   ".jpg": 0.20, ".jpeg": 0.20, ".png": 0.15,
                   ".docx": 0.20, ".pdf": 0.15},
    "Legal":      {".tiff": 0.50, ".tif": 0.50,
                   ".pdf": 0.15, ".docx": 0.15},  # ← bajar a 0.15
    "Proyectos": {".mpp": 0.60, ".xlsx": 0.25, ".pptx": 0.20, ".docx": 0.15},
    "Otros":      {".ai": 0.55, ".psd": 0.55, ".indd": 0.55,
                   ".eps": 0.45, ".svg": 0.35,
                   ".pptx": 0.20, ".pdf": 0.10},
}

CONFIDENCE_THRESHOLD = 0.15

# ─── Sistema de aprendizaje ───────────────────────────────────────────────────

FEEDBACK_FILE  = Path("feedback.json")   # fallback local
MODEL_FILE     = Path("nb_model.pkl")    # fallback local

# ── Conexión y persistencia en Postgres ───────────────────────────────────────

db_pool = None

def init_pool():
    global db_pool
    if DATABASE_URL:
        db_pool = pg_pool.SimpleConnectionPool(1, 10, DATABASE_URL, connect_timeout=5)
        logger.info("✅ Pool de conexiones creado")

def get_db_conn():
    if db_pool:
        return db_pool.getconn()
    return psycopg2.connect(DATABASE_URL, connect_timeout=5)   # fallback local

def release_db_conn(conn):
    if db_pool:
        db_pool.putconn(conn)
    else:
        conn.close()

def init_db():
    if not DATABASE_URL:
        logger.warning("⚠️  Sin DATABASE_URL, usando archivos locales")
        return
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS classifier_feedback (
                        id SERIAL PRIMARY KEY,
                        filename TEXT,
                        predicted TEXT,
                        correct TEXT,
                        preview_text TEXT,
                        confidence FLOAT,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS classifier_model (
                        id INT PRIMARY KEY,
                        data BYTEA,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_feedback_predicted ON classifier_feedback(predicted);
                    CREATE INDEX IF NOT EXISTS idx_feedback_correct   ON classifier_feedback(correct);
                    CREATE INDEX IF NOT EXISTS idx_feedback_time      ON classifier_feedback(timestamp);
                """)
            conn.commit()
        finally:
            conn.close()
        logger.info("✅ DB inicializada correctamente")
    except Exception as e:
        logger.error(f"⚠️  DB init error: {e}")
    init_pool()  # ← crea el pool DESPUÉS de que las tablas existen

def save_feedback_entry(entry: dict):
    if not DATABASE_URL:
        data = load_feedback()
        data.append(entry)
        FEEDBACK_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO classifier_feedback
                    (filename, predicted, correct, preview_text, confidence)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                entry["filename"],
                entry["predicted"],
                entry["correct"],
                entry.get("preview_text", ""),
                entry.get("confidence"),
            ))
        conn.commit()
    except Exception as e:
        logger.error(f"⚠️  Error guardando feedback: {e}")
    finally:
        release_db_conn(conn)


def save_model_to_db(model_bytes: bytes):
    if not DATABASE_URL:
        MODEL_FILE.write_bytes(model_bytes)
        return
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO classifier_model (id, data, updated_at)
                VALUES (1, %s, NOW())
                ON CONFLICT (id) DO UPDATE
                    SET data = EXCLUDED.data, updated_at = NOW()
            """, (psycopg2.Binary(model_bytes),))
        conn.commit()
        logger.info("✅ Modelo guardado en DB")
    except Exception as e:
        logger.error(f"⚠️  Error guardando modelo: {e}")
    finally:
        release_db_conn(conn)

def load_model_from_db() -> bytes | None:
    if not DATABASE_URL:
        return MODEL_FILE.read_bytes() if MODEL_FILE.exists() else None
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM classifier_model WHERE id = 1")
            row = cur.fetchone()
            return bytes(row[0]) if row else None
    except Exception:
        return None
    finally:
        release_db_conn(conn)


init_db()

def load_feedback() -> list:
    if not DATABASE_URL:
        if FEEDBACK_FILE.exists():
            return json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        return []
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT filename, predicted, correct, preview_text, confidence
                FROM classifier_feedback
                ORDER BY timestamp ASC
            """)
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"⚠️  Error leyendo feedback: {e}")
        return []
    finally:
        release_db_conn(conn)

def save_feedback(entries: list):
    pass 

# ── Capa 2: dynamic_boosts ────────────────────────────────────────────────────
# Estructura: { "Facturas": 1.15, "Contratos": 0.90 }
# > 1.0 = esta categoría fue la corrección frecuente → boostear
# < 1.0 = esta categoría fue predicha mal frecuentemente → penalizar

def compute_dynamic_boosts() -> Dict[str, float]:
    entries = load_feedback()
    if not entries:
        return {}

    # Contar cuántas veces cada categoría fue corregida (predicted → correct)
    times_wrong: Dict[str, int]    = defaultdict(int)  # fue predicha mal
    times_correct: Dict[str, int]  = defaultdict(int)  # fue la corrección real

    for e in entries:
        if e["predicted"] != e["correct"]:
            times_wrong[e["predicted"]]  += 1
            times_correct[e["correct"]]  += 1

    all_cats = set(list(times_wrong.keys()) + list(times_correct.keys()))
    boosts: Dict[str, float] = {}

    for cat in all_cats:
        wrong   = times_wrong.get(cat, 0)
        correct = times_correct.get(cat, 0)
        total   = wrong + correct
        if total == 0:
            continue
        # Ratio: qué tan seguido fue la respuesta correcta vs incorrecta
        # correct/(wrong+correct) → 1.0 = siempre correcto, 0.0 = siempre mal
        ratio = correct / total
        # Mapear a boost: ratio 1.0 → 1.30, ratio 0.5 → 1.0, ratio 0.0 → 0.75
        boosts[cat] = round(0.75 + (ratio * 0.55), 3)

    return boosts

# Cargar boosts al iniciar el servidor
dynamic_boosts: Dict[str, float] = compute_dynamic_boosts()


# ── Capa 3: Naive Bayes ───────────────────────────────────────────────────────

def train_naive_bayes() -> bool:
    global NB_MODEL_CACHE  # ← AGREGAR
    entries = load_feedback()
    if len(entries) < 10:
        return False

    texts  = [f"{e['filename']} {e.get('preview_text', '')}" for e in entries]
    labels = [e["correct"] for e in entries]

    if len(set(labels)) < 2:
        return False

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=1)
    X = vectorizer.fit_transform(texts)
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X, labels)

    save_model_to_db(pickle.dumps({"vectorizer": vectorizer, "clf": clf}))
    NB_MODEL_CACHE = None  # ← AGREGAR — fuerza recarga en próximo classify
    logger.info(f"✅ Naive Bayes entrenado con {len(entries)} ejemplos")
    return True


NB_MODEL_CACHE = None

def load_nb_model():
    global NB_MODEL_CACHE
    if NB_MODEL_CACHE:
        return NB_MODEL_CACHE
    model_bytes = load_model_from_db()
    if model_bytes:
        NB_MODEL_CACHE = pickle.loads(model_bytes)
    return NB_MODEL_CACHE


def predict_naive_bayes(filename: str, preview: str) -> Dict[str, float]:
    bundle = load_nb_model()
    if not bundle:
        return {}
    try:
        vectorizer = bundle["vectorizer"]
        clf        = bundle["clf"]
        text       = normalize(f"{filename} {preview}")
        X          = vectorizer.transform([text])
        probs      = clf.predict_proba(X)[0]
        return dict(zip(clf.classes_, probs))
    except Exception:
        return {}



# ─── Reglas de clasificación ──────────────────────────────────────────────────

CATEGORY_RULES: Dict[str, dict] = {

    # ── Facturas ──────────────────────────────────────────────────────────────
    "Facturas": {
        "keywords": [
            "factura", "factura_electronica", "factura_venta",
            "factura_compra", "factura_servicio", "factura_proforma",
            "boleta", "boleta_venta", "boleta_electronica",
            "recibo", "recibo_pago", "recibo_caja", "recibo_honorarios",
            "invoice", "bill", "receipt",
            "nota_debito", "nota_credito", "nota_ajuste",
            "comprobante_pago", "comprobante_fiscal", "comprobante_egreso",
            "comprobante_ingreso", "comprobante_contable",
            "documento_equivalente", "tiquete", "ticket_caja",
            "cfdi", "dian", "sat", "sunat", "sii", "seniat",
            "afip", "sri", "tributaria",
            "facturacion_electronica", "xml_fiscal",
            "cufe", "cae", "cuf", "uuid_fiscal",
            "cotizacion", "cotizacion_comercial", "proforma",
            "oferta_comercial", "oferta_economica", "propuesta_precio",
            "lista_precios", "tarifa", "presupuesto_cliente",
            "orden_compra", "orden_pedido", "orden_servicio",
            "solicitud_cotizacion", "rfq",
            "numero_factura", "folio", "serie_factura",
            "fecha_emision", "fecha_vencimiento", "plazo_pago",
            "subtotal", "total", "total_pagar", "valor_total",
            "valor_neto", "valor_bruto", "monto_total",
            "iva", "igv", "iva_incluido", "impuesto", "retencion",
            "descuento", "descuento_comercial", "cargo_adicional",
            "forma_pago", "metodo_pago", "condicion_pago",
            "pago_contado", "pago_credito", "pago_anticipado",
            "transferencia", "transferencia_bancaria",
            "deposito", "cheque", "efectivo",
            "pse", "nequi", "daviplata", "paypal", "stripe",
            "cuenta_bancaria", "numero_cuenta", "iban", "swift",
            "referencia_pago", "comprobante_transferencia",
            "cobranza", "cobro", "mora", "cartera_vencida",
            "estado_cuenta", "saldo_pendiente", "saldo_vencido",
            "deuda", "abono", "pago_parcial", "anticipo",
            "proveedor", "vendedor", "emisor_factura",
            "cliente", "comprador", "receptor_factura",
            "razon_social", "nombre_comercial",
            "descripcion_producto", "descripcion_servicio",
            "cantidad", "precio_unitario", "linea_factura",
            "devolucion", "reembolso", "ajuste_factura",
            "anulacion_factura", "credito_a_favor",
        ],
        "weight": 1.2
    },

    # ── Contratos ─────────────────────────────────────────────────────────────
    "Contratos": {
        "keywords": [
            "contrato", "contrato_firmado", "contrato_vigente",
            "contrato_marco", "contrato_especifico",
            "acuerdo", "acuerdo_comercial", "acuerdo_marco",
            "convenio", "convenio_cooperacion", "convenio_interinstitucional",
            "memorando", "memorando_entendimiento", "mou",
            "carta_intencion", "carta_acuerdo", "carta_compromiso",
            "protocolo_acuerdo", "minuta_contrato",
            "prestacion_servicios", "contrato_servicios",
            "contrato_consultoria", "contrato_asesoria",
            "contrato_mantenimiento", "contrato_soporte",
            "outsourcing", "subcontratacion", "contrato_obra",
            "contrato_suministro", "contrato_distribucion",
            "contrato_agencia", "contrato_franquicia",
            "sla", "acuerdo_nivel_servicio", "ans",
            "arrendamiento", "contrato_arrendamiento",
            "alquiler", "contrato_alquiler", "subarrendamiento",
            "arrendador", "arrendatario", "canon_arrendamiento",
            "deposito_garantia", "contrato_comodato",
            "licencia_uso", "licencia_software", "eula",
            "cesion_derechos", "transferencia_derechos",
            "nda", "acuerdo_confidencialidad", "confidencialidad",
            "no_divulgacion", "secreto_comercial",
            "contrato_honorarios", "contrato_prestacion",
            "contrato_freelance", "contrato_independiente",
            "contrato_compraventa", "promesa_compraventa",
            "contrato_leasing", "contrato_renting",
            "acuerdo_accionistas", "pacto_accionistas",
            "joint_venture", "contrato_sociedad",
            "contratante", "contratista", "parte_contratante",
            "parte_contratada", "comitente", "mandante",
            "licenciante", "licenciatario",
            "proveedor_servicios", "receptor_servicios",
            "clausula", "clausulas", "estipulacion",
            "objeto_contrato", "alcance_contrato",
            "obligaciones", "derechos_partes", "deberes",
            "garantias_contrato", "condiciones_generales",
            "anexo_contrato", "anexo_tecnico", "adendum",
            "otorosi", "adicion_contrato", "modificacion_contrato",
            "vigencia_contrato", "plazo_contrato", "duracion_contrato",
            "renovacion_automatica", "prorroga", "vencimiento_contrato",
            "rescision", "resolucion_contrato", "terminacion_contrato",
            "terminacion_anticipada", "terminacion_unilateral",
            "incumplimiento_contrato", "penalidad", "penalizacion",
            "multa_incumplimiento", "clausula_penal",
            "indemnizacion_contrato", "fuerza_mayor",
            "firma_contrato", "firma_partes", "suscripcion",
            "celebran_contrato", "otorgar", "partes_acuerdan",
            "firma_electronica_contrato", "firma_digital",
            "arbitraje", "mediacion_contrato", "jurisdiccion",
            "valor_contrato", "precio_contrato", "monto_contrato",
            "anticipo_contrato", "pago_hito", "retencion_garantia",
        ],
        "weight": 1.2
    },

    # ── Informes ──────────────────────────────────────────────────────────────
    "Informes": {
        "keywords": [
            "informe", "informe_final", "informe_preliminar",
            "informe_ejecutivo", "informe_gerencial", "informe_directivo",
            "informe_gestion", "informe_actividades", "informe_labores",
            "informe_tecnico", "informe_operativo", "informe_administrativo",
            "informe_comercial", "informe_ventas", "informe_produccion",
            "informe_auditoria", "informe_auditoria_interna",
            "informe_auditoria_externa", "informe_revision",
            "informe_financiero", "informe_contable", "informe_fiscal",
            "informe_sostenibilidad", "informe_rsc", "informe_esg",
            "informe_riesgo", "informe_cumplimiento", "informe_regulatorio",
            "reporte", "reporte_diario", "reporte_semanal",
            "reporte_mensual", "reporte_trimestral", "reporte_anual",
            "reporte_ejecutivo", "reporte_gerencial", "reporte_ventas",
            "reporte_financiero", "reporte_operacional",
            "memoria_anual", "memoria_corporativa", "annual_report",
            "balance_social", "rendicion_cuentas",
            "balance_general", "balance_contable",
            "estado_resultados", "estado_perdidas_ganancias",
            "flujo_caja", "flujo_efectivo", "cash_flow",
            "estado_situacion_financiera", "estado_financiero",
            "presupuesto_general", "ejecucion_presupuestal",
            "cierre_contable", "conciliacion_contable",
            "declaracion_renta", "declaracion_iva",
            "auditoria", "auditoria_interna", "auditoria_externa",
            "hallazgo_auditoria", "plan_auditoria",
            "control_interno", "dictamen", "opinion_auditoria",
            "due_diligence", "revisor_fiscal",
            "analisis", "analisis_situacional", "analisis_foda",
            "foda", "dafo", "swot", "pest", "pestel",
            "analisis_competitivo", "benchmarking", "benchmark",
            "diagnostico", "diagnostico_organizacional",
            "estudio_mercado", "estudio_viabilidad",
            "analisis_costo_beneficio", "evaluacion_impacto",
            "gap_analysis", "analisis_brecha",
            "indicadores", "indicadores_gestion", "indicadores_desempeño",
            "kpi", "kpis", "okr",
            "metricas", "metricas_negocio", "metricas_operativas",
            "dashboard", "tablero_control", "cuadro_mando",
            "balanced_scorecard", "scorecard",
            "estadistica", "estadisticas", "datos_estadisticos",
            "proyeccion", "proyecciones", "forecast", "pronostico",
            "tendencia", "tendencias", 
            "cierre_mes", "cierre_trimestre", "cierre_anual",
            "cierre_fiscal", "periodo_fiscal",
            "seguimiento", "monitoreo", "avance_actividades",
            "resultado", "resultados_obtenidos", "logro",
            "cumplimiento", "cumplimiento_meta",
            "desempeño", "rendimiento", "performance",
            "comparativo", "variacion", "desviacion",
            "pipeline", "embudo_ventas", "funnel",
            "ventas_acumuladas", "cuota_ventas",
            "eficiencia_operativa", "productividad", "oee",
            "inventario", "reporte_inventario", "kardex",
            "rotacion_personal", "ausentismo_reporte",
            "informe_sarlaft", "informe_sagrilaft",
        ],
        "weight": 1.0
    },

    # ── Personal ──────────────────────────────────────────────────────────────
    "Personal": {
        "keywords": [
            "curriculum", "curriculo", "curriculum_vitae", "hoja_de_vida",
            "hoja_vida", "cv", "resume", "perfil_profesional",
            "perfil_candidato", "datos_personales", "informacion_personal",
            "experiencia_laboral", "experiencia_profesional",
            "formacion_academica", "estudios_realizados",
            "logros_profesionales", "referencias_laborales",
            "competencias", "habilidades", "aptitudes",
            "candidato", "postulante", "aspirante", "aplicante",
            "proceso_seleccion", "entrevista", "entrevista_trabajo",
            "prueba_tecnica", "prueba_psicologica", "assessment",
            "oferta_laboral", "carta_oferta", "propuesta_laboral",
            "empleado", "trabajador", "funcionario", "colaborador",
            "staff", "planta_personal", "headcount",
            "ficha_empleado", "legajo", "expediente_empleado",
            "contrato_laboral", "contrato_trabajo", "contrato_indefinido",
            "contrato_fijo", "contrato_temporal", "renovacion_contrato",
            "nomina", "payroll", "liquidacion_sueldo", "liquidacion_salario",
            "recibo_sueldo", "recibo_nomina", "comprobante_nomina",
            "colilla_pago", "desprendible_pago",
            "salario", "sueldo", "remuneracion", "salario_base",
            "bonificacion", "bono", "comision", "incentivo",
            "prestaciones_sociales", "prima", "cesantias",
            "aguinaldo", "decimotercer_mes", "beneficios_empleado",
            "aumento_sueldo", "ajuste_salarial",
            "renuncia", "carta_renuncia", "dimision",
            "carta_despido", "despido",
            "desvinculacion", "liquidacion_laboral", "finiquito",
            "acta_liquidacion", "indemnizacion_laboral",
            "cedula", "cedula_ciudadania", "cedula_identidad",
            "dni", "documento_identidad", "pasaporte",
            "tarjeta_identidad", "carnet", "rut_persona",
            "foto_empleado", "fotografia_personal", "foto_perfil",
            "foto_cedula", "foto_carnet", "foto_documento",
            "foto_profesional", "headshot", "retrato",
            "vacaciones", "solicitud_vacaciones", "periodo_vacacional",
            "licencia", "licencia_medica", "licencia_maternidad",
            "licencia_paternidad", "permiso", "permiso_ausencia",
            "incapacidad", "incapacidad_medica", "baja_medica",
            "evaluacion_desempeño", "evaluacion_empleado",
            "revision_desempeño", "plan_desarrollo", "plan_carrera",
            "capacitacion", "formacion", "entrenamiento",
            "onboarding", "induccion", "certificacion_empleado",
            "seguridad_social", "eps", "arl", "afp", "pension",
            "caja_compensacion", "afiliacion_eps", "afiliacion_pension",
            "historia_clinica", "examen_medico", "certificado_medico",
            "aptitud_laboral",
            "reglamento_interno", "reglamento_trabajo",
            "politica_rrhh", "manual_empleado", "manual_bienvenida",
            "organigrama", "estructura_organizacional",
            "directorio_empleados", "cargo", "puesto", "descripcion_cargo",
        ],
        "weight": 1.0
    },

    # ── Legal ─────────────────────────────────────────────────────────────────
    "Legal": {
        "keywords": [
            "escritura_publica", "escritura_notarial", "escritura_aclaratoria",
            "minuta_notarial", "poder_notarial", "poder_especial",
            "poder_general", "revocacion_poder", "apoderado",
            "notarial", "notaria", "notario", "fe_publica",
            "autenticacion", "apostilla", "legalizacion",
            "protocolizacion", "elevacion_escritura",
            "acta_constitucion", "acta_constitutiva", "acta_fundacion",
            "estatutos_sociales", "estatutos_empresa", "reglamento_estatutario",
            "registro_mercantil", "matricula_mercantil", "renovacion_matricula",
            "camara_comercio", "certificado_existencia",
            "rut_empresa", "nit_empresa", "personeria_juridica",
            "objeto_social", "capital_social", "socios_fundadores",
            "representante_legal", "reforma_estatutos",
            "liquidacion_empresa", "disolucion_sociedad",
            "acta_asamblea", "acta_junta_directiva", "acta_junta_socios",
            "acta_extraordinaria", "acta_ordinaria",
            "acta_nombramiento", "acta_remocion", "libro_actas",
            "demanda", "demanda_civil", "demanda_laboral", "demanda_penal",
            "contestacion_demanda", "excepcion_previa",
            "recurso", "recurso_apelacion", "recurso_casacion",
            "tutela", "accion_tutela", "habeas_corpus",
            "medida_cautelar", "embargo", "secuestro_bien",
            "sentencia", "sentencia_judicial", "fallo",
            "auto", "providencia", "decision_judicial",
            "resolucion_judicial", "resolucion_administrativa",
            "juzgado", "tribunal", "corte", "consejo_estado",
            "expediente", "expediente_judicial", "numero_radicado",
            "notificacion_judicial", "emplazamiento", "citacion",
            "conciliacion_judicial", "laudo_arbitral",
            "ley_nacional", "codigo_civil",
            "codigo_comercio", "codigo_penal", "codigo_laboral",
            "decreto", "decreto_ley", "decreto_reglamentario",
            "resolucion_normativa", "circular_legal",
            "directiva", "ordenanza", "acuerdo_municipal",
            "normativa", "normatividad", "regulacion_legal",
            "marco_legal", "marco_normativo", "cumplimiento_legal",
            "concepto_juridico", "concepto_legal",
            "patente", "solicitud_patente", "registro_patente",
            "marca_registrada", "registro_marca",
            "derechos_autor", "derecho_autor", "copyright",
            "propiedad_intelectual", "propiedad_industrial",
            "proteccion_datos", "habeas_data",
            "consentimiento_datos", "tratamiento_datos",
            "dpo", "delegado_proteccion",
            "sarlaft", "sagrilaft", "lavado_activos",
            "debida_diligencia", "due_diligence_legal",
            "lista_ofac", "lista_clinton",
            "gobierno_corporativo", "codigo_buen_gobierno",
            "fianza", "garantia_legal", "hipoteca", "prenda", "gravamen",
            "fideicomiso", "patrimonio_autonomo",
            "denuncia_penal", "querella", "investigacion_penal",
            "sancion_disciplinaria", "proceso_disciplinario",
            "pliego_cargos", "descargos", "fallo_disciplinario",
            "paz_salvo", "paz_y_salvo", "antecedentes",
            "certificado_judicial", "certificado_disciplinario",
            "declaracion_juramentada", "opinion_legal",
        ],
        "weight": 1.2
    },

    # ── Proyectos ─────────────────────────────────────────────────────────────
    "Proyectos": {
        "keywords": [
            "acta_proyecto", "acta_inicio", "acta_constitutiva_proyecto",
            "project_charter", "project_brief", "kick_off",
            "alcance_proyecto", "scope", "scope_statement",
            "objetivo_proyecto", "justificacion_proyecto",
            "caso_negocio", "business_case", "viabilidad_proyecto",
            "terminos_referencia", "tor", "pliego_condiciones",
            "cronograma", "cronograma_proyecto", "cronograma_actividades",
            "plan_trabajo", "plan_proyecto", "plan", "plan_ejecucion",
            "plan_gestion_proyecto", "plan_director",
            "gantt", "diagrama_gantt", "carta_gantt",
            "ruta_critica", "camino_critico", "critical_path",
            "pert", "diagrama_pert", "edt", "wbs",
            "estructura_desglose", "work_breakdown",
            "linea_base", "baseline", "holgura", "float",
            "hito", "hitos", "milestone", "milestones",
            "entregable", "entregables", "deliverable", "deliverables",
            "criterio_aceptacion", "acta_entrega", "acta_aceptacion",
            "agile", "agilidad", "metodologia_agil",
            "scrum", "scrum_master", "product_owner",
            "sprint", "sprint_planning", "sprint_review",
            "sprint_retrospective", "sprint_backlog",
            "daily", "daily_scrum", "standup",
            "backlog", "product_backlog", "refinamiento",
            "historia_usuario", "user_story", "epic",
            "definition_of_done", "burndown", "burnup",
            "kanban", "tablero_kanban", "wip", "work_in_progress",
            "safe", "scaled_agile",
            "waterfall", "cascada", "pmbok", "pmi", "pmp",
            "prince2", "ipma", "iso_21500",
            "gestor_proyecto", "director_proyecto", "project_manager",
            "pmo", "oficina_proyectos",
            "stakeholder", "interesado", "parte_interesada",
            "registro_interesados", "sponsor", "patrocinador_proyecto",
            "equipo_proyecto", "raci", "matriz_raci",
            "riesgo_proyecto", "gestion_riesgos",
            "plan_riesgos", "registro_riesgos", "matriz_riesgos",
            "mitigacion_riesgo", "plan_contingencia",
            "incidencia", "issue", "registro_issues",
            "control_cambios", "gestion_cambios",
            "solicitud_cambio", "change_request",
            "comite_cambios", "ccb", "log_cambios",
            "presupuesto_proyecto", "budget_proyecto",
            "estimacion_costos", "valor_ganado", "earned_value",
            "cpi", "spi", "variacion_costo", "variacion_cronograma",
            "plan_calidad_proyecto", "control_calidad_proyecto", "uat",
            "plan_comunicaciones", "informe_avance_proyecto",
            "status_report", "dashboard_proyecto",
            "minuta_reunion", "acta_reunion_proyecto",
            "plan_adquisiciones", "gestion_adquisiciones",
            "cierre_proyecto", "acta_cierre", "informe_cierre",
            "lecciones_aprendidas", "lessons_learned",
            "retrospectiva", "post_mortem", "evaluacion_proyecto",
        ],
        "weight": 1.0
    },

    # ── Otros ─────────────────────────────────────────────────────────────────
    "Otros": {
        "keywords": [
            "manual", "manual marca", "manual de marca", "manual_usuario", "manual_operacion",
            "manual_instalacion", "guia de marca", "manual_configuracion",
            "manual_administrador", "manual_tecnico",
            "manual_procedimientos", "brandbook marca", "manual_calidad",
            "guia", "guia_usuario", "guia_rapida", "guia_referencia",
            "getting_started", "quick_guide",
            "instructivo", "instructivo_uso", "instrucciones",
            "tutorial", "paso_a_paso", "how_to", "walkthrough",
            "procedimiento", "procedimiento_operativo",
            "procedimiento_estandar", "sop",
            "protocolo", "protocolo_atencion", "protocolo_seguridad",
            "politica", "politica_interna", "politica_empresarial",
            "politica_uso", "politica_seguridad", "politica_tic",
            "politica_ambiental", "politica_calidad",
            "terminos_condiciones", "terminos_uso", "terminos_servicio",
            "reglamento", "normas_internas", "lineamiento",
            "codigo_conducta", "codigo_etica_corporativo",
            "manual_marca", "manual_identidad", "brandbook",
            "brand_guidelines", "identidad_corporativa", "identidad_visual",
            "imagen_corporativa", "branding", "rebranding",
            "logotipo", "logo", "isotipo", "imagotipo", "isologo",
            "paleta_colores", "tipografia_corporativa",
            "patron_grafico", "tono_comunicacion", "voz_marca",
            "brochure", "folleto", "flyer", "triptico", "diptico",
            "catalogo", "catalogo_productos", "catalogo_servicios",
            "ficha_producto", "ficha_tecnica_producto",
            "presentacion_comercial", "presentacion_empresa",
            "presentacion_corporativa", "company_profile",
            "pitch", "pitch_deck", "investor_deck",
            "portafolio", "portafolio_servicios",
            "propuesta_comercial", "propuesta_valor",
            "one_pager", "fact_sheet", "infografia",
            "banner", "afiche", "poster", "cartel", "volante",
            "material_pop", "kit_prensa", "press_kit", "media_kit",
            "newsletter", "boletin", "comunicado_prensa", "nota_prensa",
            "plan_marketing", "estrategia_marketing",
            "campana_marketing", "campana_publicitaria",
            "comunicado_interno", "circular_interna", "memo_interno",
            "anuncio_empresa", "comunicacion_corporativa",
            "carta_bienvenida", "carta_presentacion",
            "carta_referencia", "carta_recomendacion",
            "invitacion", "invitacion_evento", "convocatoria",
            "presentacion", "diapositivas", "slides", "deck",
            "plantilla", "template", "formato",
            "formulario", "formato_solicitud", "formato_reporte",
            "check_list", "lista_verificacion", "lista_chequeo",
            "encuesta", "cuestionario", "formulario_encuesta",
            "documentacion_tecnica", "especificacion_tecnica",
            "especificacion_software", "requerimientos_sistema",
            "arquitectura_sistema", "diagrama_sistema",
            "diagrama_flujo", "flujograma",
            "readme", "changelog", "release_notes",
            "api_documentation", "swagger",
            "agenda", "agenda_evento", "programa_evento",
            "logistica_evento", "plan_evento",
            "foto_producto", "foto_instalaciones", "foto_evento",
            "foto_corporativa", "captura_pantalla", "screenshot",
        ],
        "weight": 0.7
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[_\-\./\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_spacy_tokens(raw_text: str) -> List[str]:
    if not nlp:
        return []
    doc = nlp(raw_text[:400])
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token.text) > 3
    ]


def build_ngrams(tokens: List[str], n: int) -> List[str]:
    """MEJORA 6: soporta bigramas y trigramas."""
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def keyword_in_text(kw_norm: str, text: str) -> bool:
    """Busca keyword como palabra completa, no como substring."""
    return bool(re.search(r'(?<!\w)' + re.escape(kw_norm) + r'(?!\w)', text))

def calculate_confidence(
    filename: str,
    mime_type: str,
    preview: str
) -> Tuple[str, float, List[str]]:

    # ── 1. SpaCy ──────────────────────────────────────────────────────────────
    raw_combined = f"{filename} {preview or ''}"
    spacy_tokens = extract_spacy_tokens(raw_combined)

    # ── 2. Normalizar + ngrams ────────────────────────────────────────────────
    norm_filename = normalize(filename)
    norm_preview  = normalize(preview or "")
    spacy_str     = " ".join(spacy_tokens)

    filename_tokens = norm_filename.split()
    preview_tokens  = norm_preview.split()

    # MEJORA 6: bigramas + trigramas
    enriched_filename = " ".join(
        [norm_filename]
        + build_ngrams(filename_tokens, 2)
        + build_ngrams(filename_tokens, 3)
    )
    enriched_content = " ".join(
        [norm_preview, spacy_str]
        + build_ngrams(preview_tokens, 2)
        + build_ngrams(preview_tokens, 3)
    )

    ext = os.path.splitext(filename)[1].lower()

    scores: Dict[str, float]     = {}
    all_matched: Dict[str, list] = {}

    for cat, rules in CATEGORY_RULES.items():
        score    = 0.0
        matched  = []
        keywords = rules["keywords"]
        total_kw = len(keywords)
        w        = rules["weight"]

        # MEJORA 1: normalizar por densidad (1 / total_keywords)
        # evita que categorías con más keywords ganen por volumen puro
        for kw in keywords:
            kw_norm = normalize(kw)
            if keyword_in_text(kw_norm, enriched_filename):
                score += (1.0 / total_kw) * w * 3.0
                matched.append(f"filename:{kw}")
            elif keyword_in_text(kw_norm, enriched_content):
                score += (1.0 / total_kw) * w * 1.5
                matched.append(f"content:{kw}")

        # MEJORA 5: peso dinámico por extensión
        ext_w = EXTENSION_WEIGHTS.get(cat, {}).get(ext, 0)
        if ext_w > 0:
            effective_ext_w = ext_w if matched else ext_w * 0.4
            score += effective_ext_w
            matched.append(f"ext:{ext}(+{effective_ext_w})")

        scores[cat]      = round(score, 4)
        all_matched[cat] = matched

    # ── Capa 2: aplicar dynamic_boosts del feedback ───────────────────────────
    for cat in scores:
        boost = dynamic_boosts.get(cat, 1.0)
        if boost != 1.0:
            scores[cat] = round(scores[cat] * boost, 4)

    # ── Capa 3: fusionar con Naive Bayes si hay modelo entrenado ──────────────
    nb_probs = predict_naive_bayes(filename, preview)
    if nb_probs:
        for cat in scores:
            nb_score    = nb_probs.get(cat, 0.0)
            scores[cat] = round(scores[cat] * 0.80 + nb_score * 0.20, 4)

    # ── 3. Selección con score relativo ───────────────────────────────────────
    if not any(s > 0 for s in scores.values()):
        return "Sin clasificar", 0.0, []

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = sorted_scores[0]
    _, second_score    = sorted_scores[1] if len(sorted_scores) > 1 else ("", 0.0)

    # MEJORA 4: confianza relativa = qué tan superior es el ganador
    relative_gap = top_score - second_score
    if top_score > 0:
        normalized_gap = min(relative_gap / max(top_score, 0.01), 1.0)
    else:
        normalized_gap = 0.0

    # 70% score absoluto + 30% margen sobre el segundo
    abs_normalized   = min(top_score, 1.0)
    final_confidence = round((abs_normalized * 0.70) + (normalized_gap * 0.30), 4)

    # MEJORA 1: penalizar "Otros" si otra categoría está muy cerca
    if top_cat == "Otros" and len(sorted_scores) > 1:
        second_cat, second_sc = sorted_scores[1]
        if second_sc >= top_score * 0.85:
            top_cat = second_cat
            final_confidence = round(final_confidence * 0.80, 4)

    if final_confidence < CONFIDENCE_THRESHOLD:
        return "Sin clasificar", final_confidence, []

    return top_cat, final_confidence, all_matched.get(top_cat, [])


# ─── Endpoints ────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    filename: str
    predicted: str        # lo que dijo la IA
    correct: str          # lo que el usuario corrigió
    preview_text: Optional[str] = None
    confidence: Optional[float] = None


class FeedbackStats(BaseModel):
    total_feedback: int
    corrections: int          # predicted != correct
    accuracy: float           # % de veces que acertó
    most_confused: list       # pares [predicted, correct] más frecuentes
    boosts: Dict[str, float]  # boosts actuales por categoría


@app.post("/feedback", status_code=201)
async def submit_feedback(req: FeedbackRequest, bg: BackgroundTasks):
    global dynamic_boosts

    req.filename     = (req.filename or "")[:255]       
    req.preview_text = (req.preview_text or "")[:2000]  

    save_feedback_entry({
        "filename":     req.filename,
        "predicted":    req.predicted,
        "correct":      req.correct,
        "preview_text": req.preview_text or "",
        "confidence":   req.confidence,
    })
    entries = load_feedback()
    dynamic_boosts = compute_dynamic_boosts()

    corrections = [e for e in entries if e["predicted"] != e["correct"]]
    if len(corrections) % 10 == 0 and len(corrections) > 0:
        bg.add_task(train_naive_bayes)  # ← no bloquea el request

    return {
        "saved": True,
        "total_feedback": len(entries),
        "boosts_updated": dynamic_boosts,
    }



@app.get("/feedback/stats", response_model=FeedbackStats)
async def feedback_stats():
    """Panel de diagnóstico: ver qué tan bien está clasificando."""
    entries     = load_feedback()
    corrections = [e for e in entries if e["predicted"] != e["correct"]]
    accuracy    = round(1 - len(corrections) / max(len(entries), 1), 3)

    # Top pares de confusión
    confusion: Dict[str, int] = defaultdict(int)
    for e in corrections:
        key = f"{e['predicted']} → {e['correct']}"
        confusion[key] += 1

    most_confused = sorted(
        [{"pair": k, "count": v} for k, v in confusion.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:5]

    return FeedbackStats(
        total_feedback=len(entries),
        corrections=len(corrections),
        accuracy=accuracy,
        most_confused=most_confused,
        boosts=dynamic_boosts,
    )


@app.post("/retrain")
async def retrain():
    """Fuerza re-entrenamiento del modelo Naive Bayes manualmente."""
    global dynamic_boosts
    dynamic_boosts = compute_dynamic_boosts()
    success = train_naive_bayes()
    entries = load_feedback()
    return {
        "trained": success,
        "total_samples": len(entries),
        "boosts": dynamic_boosts,
        "message": "OK" if success else "Mínimo 10 ejemplos con 2+ categorías para entrenar NB",
    }

@app.post("/classify", response_model=ClassifyResponse)
async def classify_document(request: ClassifyRequest) -> ClassifyResponse:
    request.file_name    = (request.file_name or "")[:255]      
    request.preview_text = (request.preview_text or "")[:2000] 

    category, confidence, rules_matched = calculate_confidence(
        request.file_name,
        request.mime_type or "",
        request.preview_text or ""
    )

    suggested_tags = (
        [category.lower().replace(" ", "_")]
        if category != "Sin clasificar"
        else []
    )

    return ClassifyResponse(
        category=category,
        confidence=confidence,
        suggested_tags=suggested_tags,
        rules_matched=rules_matched
    )


@app.get("/health")
async def health():
    return {
        "status":  "healthy",
        "spaCy":   nlp is not None,
        "model":   "es_core_news_sm" if nlp else "none",
        "version": "2.0"
    }


@app.get("/categories")
async def get_categories():
    return list(CATEGORY_RULES.keys()) + ["Sin clasificar"]


@app.post("/debug")
async def debug_classify(request: ClassifyRequest):
    """Devuelve scores de TODAS las categorías para diagnóstico."""
    if os.getenv("ENV", "production") != "development":
        raise HTTPException(status_code=403, detail="Solo disponible en desarrollo")
    from fastapi.responses import JSONResponse

    norm_filename = normalize(request.file_name)
    norm_preview  = normalize(request.preview_text or "")
    spacy_tokens  = extract_spacy_tokens(f"{request.file_name} {request.preview_text or ''}")
    spacy_str     = " ".join(spacy_tokens)

    filename_tokens = norm_filename.split()
    preview_tokens  = norm_preview.split()

    enriched_filename = " ".join(
        [norm_filename] + build_ngrams(filename_tokens, 2) + build_ngrams(filename_tokens, 3)
    )
    enriched_content = " ".join(
        [norm_preview, spacy_str] + build_ngrams(preview_tokens, 2) + build_ngrams(preview_tokens, 3)
    )
    ext = os.path.splitext(request.file_name)[1].lower()

    debug_scores = {}
    for cat, rules in CATEGORY_RULES.items():
        score, matched = 0.0, []
        total_kw = len(rules["keywords"])
        w = rules["weight"]
        for kw in rules["keywords"]:
            kw_norm = normalize(kw)
            if keyword_in_text(kw_norm, enriched_filename):
                score += (1.0 / total_kw) * w * 3.0
                matched.append(f"filename:{kw}")
            elif keyword_in_text(kw_norm, enriched_content):
                score += (1.0 / total_kw) * w * 1.5
                matched.append(f"content:{kw}")
        ext_w = EXTENSION_WEIGHTS.get(cat, {}).get(ext, 0)
        if ext_w:
            effective_ext_w = ext_w if matched else ext_w * 0.4
            score += effective_ext_w
            matched.append(f"ext:{ext}(+{effective_ext_w})")
        debug_scores[cat] = {"score": round(score, 4), "matched": matched}

    sorted_debug = dict(
        sorted(debug_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    )
    winner, conf, _ = calculate_confidence(
        request.file_name, request.mime_type or "", request.preview_text or ""
    )

    return JSONResponse({
        "winner": winner,
        "final_confidence": conf,
        "all_scores": sorted_debug,
        "spacy_tokens": spacy_tokens,
        "extension": ext,
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    is_dev = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=is_dev)