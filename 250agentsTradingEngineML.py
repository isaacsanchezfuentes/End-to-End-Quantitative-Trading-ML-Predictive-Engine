import datetime
import json
import math
import os
import time
import warnings
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==========================================
# 0. MOTORES DE PROBABILIDAD PONDERADA ICT
# ==========================================
class ICTProbabilityEngine:
    def __init__(self, weight_ict1=0.40, weight_ict2=0.40, min_threshold=0.52):
        self.w_ict1 = weight_ict1
        self.w_ict2 = weight_ict2
        self.min_threshold = min_threshold

    def calculate_probability(
        self,
        ict1_structure: bool,
        ict2_liquidity: bool,
        candle_body: float,
        avg_body: float,
        is_exiting_zone: bool,
    ) -> dict:

        score_ict1 = self.w_ict1 if ict1_structure else 0.0
        score_ict2 = self.w_ict2 if ict2_liquidity else 0.0
        score_acc = 0.0

        if is_exiting_zone:
            if candle_body >= (avg_body * 2.0):
                score_acc = 0.15
            elif candle_body >= (avg_body * 1.5):
                score_acc = 0.05

        raw_prob = score_ict1 + score_ict2 + score_acc
        final_prob = min(1.0, round(raw_prob, 2))

        return {
            "probability": final_prob,
            "triggered": final_prob >= self.min_threshold,
        }

ENGINE_ICT_BUY = ICTProbabilityEngine(min_threshold=0.52)
ENGINE_ICT_SELL = ICTProbabilityEngine(min_threshold=0.52)

# ==========================================
# 1. PARÁMETROS Y CONFIGURACIÓN
# ==========================================
VERSION_SISTEMA = "v10.9-V5-PERFECTED-HARD-FUSE"
VERSION_TAG = "v9.4.0"
SYMBOL = "XAUUSD+"

MODELO_ICT_BUY_PATH = "cerebro_wprICT_BUY.json"
MODELO_SESIONES_BUY_PATH = "cerebro_wprICTsesiones_BUY.json"
MODELO_ICT_SELL_PATH = "cerebro_wprICT_SELL.json"
MODELO_SESIONES_SELL_PATH = "cerebro_wprICTsesiones_SELL.json"

CARPETA_MODELOS_BUY = "agentes_especializados_buy"
CARPETA_MODELOS_SELL = "agentes_especializados_sell"

LOTE_FIJO = 0.01

MAGIC_BUY = 888111
MAGIC_SELL = 888999
TAMANO_NICHO = 50

PICO_MAXIMO_BUY = 0.0
PICO_MAXIMO_SELL = 0.0
TIEMPO_ENTRADA_BUY = 0
TIEMPO_ENTRADA_SELL = 0

TIEMPO_RESPIRO_SEGUNDOS = 90
FACTOR_COLCHON_ATR = 0.30

TZ_MEXICO = pytz.timezone("America/Mexico_City")
TZ_UTC = pytz.utc
ARCHIVO_LOG = "auditoria_oro_elite_dual.log"
ARCHIVO_TELEMETRIA = "telemetria_dashboard.json"

for c in [CARPETA_MODELOS_BUY, CARPETA_MODELOS_SELL]:
    if not os.path.exists(c):
        os.makedirs(c)

SISTEMA_TRADES_TOTALES = 0
SHADOW_TRADES_TOTALES = 0
ULTIMO_CONTEO_GUARDADO = 0

REGIMENES = ["TOKYO", "LONDON", "NY", "OVERLAP", "OTROS"]

BOUNDS_POR_REGIMEN = {
    "TOKYO": {"score_minimo": (0.52, 0.92), "wpr_exit": (-85.0, -15.0), "reversal_score_limit": (0.45, 0.52), "ratio_mecha_exit": (0.60, 0.85), "anomalia_vol_exit": (2.5, 6.0), "vel_freno_exit": (0.1, 1.0), "agresividad_lote": (0.8, 2.0), "retroceso_pico_exit": (0.20, 0.50), "vel_regreso_tolerada": (0.3, 2.5)},
    "LONDON": {"score_minimo": (0.52, 0.90), "wpr_exit": (-88.0, -12.0), "reversal_score_limit": (0.38, 0.46), "ratio_mecha_exit": (0.40, 0.70), "anomalia_vol_exit": (1.2, 4.0), "vel_freno_exit": (0.3, 1.8), "agresividad_lote": (1.2, 3.5), "retroceso_pico_exit": (0.25, 0.55), "vel_regreso_tolerada": (0.5, 3.0)},
    "NY": {"score_minimo": (0.52, 0.84), "wpr_exit": (-88.0, -12.0), "reversal_score_limit": (0.40, 0.48), "ratio_mecha_exit": (0.45, 0.75), "anomalia_vol_exit": (1.5, 4.5), "vel_freno_exit": (0.2, 1.5), "agresividad_lote": (1.0, 3.2), "retroceso_pico_exit": (0.20, 0.50), "vel_regreso_tolerada": (0.4, 2.8)},
    "OVERLAP": {"score_minimo": (0.52, 0.85), "wpr_exit": (-90.0, -10.0), "reversal_score_limit": (0.42, 0.50), "ratio_mecha_exit": (0.50, 0.80), "anomalia_vol_exit": (1.8, 5.0), "vel_freno_exit": (0.2, 1.4), "agresividad_lote": (1.0, 3.0), "retroceso_pico_exit": (0.20, 0.45), "vel_regreso_tolerada": (0.6, 3.5)},
    "OTROS": {"score_minimo": (0.55, 0.99), "wpr_exit": (-85.0, -15.0), "reversal_score_limit": (0.42, 0.50), "ratio_mecha_exit": (0.55, 0.80), "anomalia_vol_exit": (2.0, 5.5), "vel_freno_exit": (0.1, 1.2), "agresividad_lote": (0.8, 2.5), "retroceso_pico_exit": (0.25, 0.50), "vel_regreso_tolerada": (0.3, 2.5)},
}

# ==========================================
# 2. AGENTE GENÉTICO UNIFICADO
# ==========================================
class EliteAgent:
    def __init__(self, agent_id, regimen, direccion="BUY"):
        self.id = str(agent_id)
        self.regimen = str(regimen)
        self.direccion = direccion
        try: idx = int(str(agent_id).split("_")[-1])
        except (ValueError, IndexError): idx = 0
        self.usar_wpr_5m = bool(idx >= 25)
        self.en_mercado = False
        self.precio_entrada_virtual = 0.0
        self.pico_maximo_pnl = 0.0
        self.fitness_score = 0.0
        self.shadow_pnl = 0.0
        self.trades_totales = 0
        self.trades_ganados = 0
        self.reiniciar_genes()

    def reiniciar_genes(self):
        bounds = BOUNDS_POR_REGIMEN.get(self.regimen, BOUNDS_POR_REGIMEN["OTROS"])
        self.score_minimo = float(np.random.uniform(*bounds["score_minimo"]))
        self.wpr_exit = float(np.random.uniform(*bounds["wpr_exit"]))
        self.reversal_score_limit = float(np.random.uniform(*bounds["reversal_score_limit"]))
        self.ratio_mecha_exit = float(np.random.uniform(*bounds["ratio_mecha_exit"]))
        self.anomalia_vol_exit = float(np.random.uniform(*bounds["anomalia_vol_exit"]))
        self.vel_freno_exit = float(np.random.uniform(*bounds["vel_freno_exit"]))
        self.agresividad_lote = float(np.random.uniform(*bounds["agresividad_lote"]))
        self.retroceso_pico_exit = float(np.random.uniform(*bounds["retroceso_pico_exit"]))
        self.vel_regreso_tolerada = float(np.random.uniform(*bounds["vel_regreso_tolerada"]))

    def evaluar_shadow_trade(self, precio_actual, score_actual, fisica, atr, permiso_entrada=True):
        trade_cerrado = False
        if not self.en_mercado:
            if permiso_entrada and (score_actual >= self.score_minimo):
                self.en_mercado, self.precio_entrada_virtual, self.pico_maximo_pnl = True, precio_actual, 0.0
        else:
            pnl_flotante = (precio_actual - self.precio_entrada_virtual) if self.direccion == "BUY" else (self.precio_entrada_virtual - precio_actual)
            if pnl_flotante > self.pico_maximo_pnl: self.pico_maximo_pnl = pnl_flotante
            wpr_eval = fisica["wpr_5m"] if self.usar_wpr_5m else fisica["wpr"]
            cierre_virtual = False

            if pnl_flotante > 0:
                if pnl_flotante <= (self.pico_maximo_pnl * (1.0 - self.retroceso_pico_exit)): cierre_virtual = True
                elif (self.direccion == "BUY" and (wpr_eval >= self.wpr_exit or score_actual < self.reversal_score_limit)) or \
                     (self.direccion == "SELL" and (wpr_eval <= self.wpr_exit or score_actual < self.reversal_score_limit)): cierre_virtual = True
                elif (self.direccion == "BUY" and fisica["mecha_sup"] >= self.ratio_mecha_exit) or \
                     (self.direccion == "SELL" and fisica["mecha_inf"] >= self.ratio_mecha_exit): cierre_virtual = True
            if pnl_flotante <= -(2.0 * atr): cierre_virtual = True

            if cierre_virtual:
                self.en_mercado = False
                self.trades_totales += 1
                if pnl_flotante > 0: self.trades_ganados += 1
                self.fitness_score += pnl_flotante
                self.shadow_pnl = round(self.fitness_score, 2)
                trade_cerrado = True
        return trade_cerrado

NICHOS_BUY = {r: [EliteAgent(f"BUY_{r}_{i}", r, "BUY") for i in range(TAMANO_NICHO)] for r in REGIMENES}
NICHOS_SELL = {r: [EliteAgent(f"SELL_{r}_{i}", r, "SELL") for i in range(TAMANO_NICHO)] for r in REGIMENES}
LIDERES_BUY = {r: NICHOS_BUY[r][0] for r in REGIMENES}
LIDERES_SELL = {r: NICHOS_SELL[r][0] for r in REGIMENES}

# ==========================================
# 3. SISTEMA DE LOGGING ESTRUCTURADO
# ==========================================
def auditar_calculo(evento: str, regimen="N/A", score=None, etiqueta="[INFO]", datos_mt5: dict = None):
    now_utc = datetime.datetime.now(TZ_UTC)
    now_mx = now_utc.astimezone(TZ_MEXICO)
    record = {
        "time_mx": now_mx.strftime("%Y-%m-%d %H:%M:%S"), "time_mt5": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "etiqueta": etiqueta, "version_sistema": VERSION_SISTEMA, "regimen": regimen, "evento": evento,
        "conviccion_xgb": round(score * 100, 2) if score is not None else None, "ticket_mt5": None,
        "order_mt5": None, "agente_id": None, "symbol": SYMBOL, "tipo_op": None, "volumen": None,
        "precio_teorico": None, "precio_ejecutado": None, "slippage_pts": None, "comision": None,
        "swap": None, "profit_bruto": None, "profit_neto": None, "motivo_cierre": None,
    }
    if datos_mt5: record.update(datos_mt5)
    json_line = json.dumps(record, default=str)
    print(f"[{record['time_mx']} MX] | {etiqueta} | [{regimen}] | {evento}", flush=True)
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as f: f.write(json_line + "\n")

def actualizar_lideres_y_evolucion():
    global LIDERES_BUY, LIDERES_SELL
    for r in REGIMENES:
        NICHOS_BUY[r].sort(key=lambda a: a.fitness_score, reverse=True)
        nuevo_lider_buy = NICHOS_BUY[r][0]
        antiguo_lider_buy = LIDERES_BUY.get(r, nuevo_lider_buy)
        if nuevo_lider_buy.id != antiguo_lider_buy.id:
            score_suavizado = (antiguo_lider_buy.score_minimo + nuevo_lider_buy.score_minimo) / 2.0
            nuevo_lider_buy.score_minimo = float(round(score_suavizado, 4))
        LIDERES_BUY[r] = nuevo_lider_buy

        NICHOS_SELL[r].sort(key=lambda a: a.fitness_score, reverse=True)
        nuevo_lider_sell = NICHOS_SELL[r][0]
        antiguo_lider_sell = LIDERES_SELL.get(r, nuevo_lider_sell)
        if nuevo_lider_sell.id != antiguo_lider_sell.id:
            score_suavizado = (antiguo_lider_sell.score_minimo + nuevo_lider_sell.score_minimo) / 2.0
            nuevo_lider_sell.score_minimo = float(round(score_suavizado, 4))
        LIDERES_SELL[r] = nuevo_lider_sell

def guardar_poblacion_agentes():
    fecha_actual = datetime.datetime.now(TZ_MEXICO).strftime("%Y-%m-%d %H:%M:%S")
    for regimen in REGIMENES:
        data_buy = {"metadata": {"version_sistema": VERSION_SISTEMA, "version_tag": VERSION_TAG, "direccion": "BUY", "regimen": regimen, "fecha_actualizacion": fecha_actual}, "agentes": [a.__dict__ for a in NICHOS_BUY[regimen]]}
        with open(os.path.join(CARPETA_MODELOS_BUY, f"agentes_{regimen}_{VERSION_TAG}.json"), "w", encoding="utf-8") as f: json.dump(data_buy, f, indent=4)
        data_sell = {"metadata": {"version_sistema": VERSION_SISTEMA, "version_tag": VERSION_TAG, "direccion": "SELL", "regimen": regimen, "fecha_actualizacion": fecha_actual}, "agentes": [a.__dict__ for a in NICHOS_SELL[regimen]]}
        with open(os.path.join(CARPETA_MODELOS_SELL, f"agentes_{regimen}_{VERSION_TAG}.json"), "w", encoding="utf-8") as f: json.dump(data_sell, f, indent=4)

def cargar_poblacion_agentes():
    for regimen in REGIMENES:
        path_buy = os.path.join(CARPETA_MODELOS_BUY, f"agentes_{regimen}_{VERSION_TAG}.json")
        if os.path.exists(path_buy):
            try:
                with open(path_buy, "r", encoding="utf-8") as f:
                    for idx, a_dict in enumerate(json.load(f).get("agentes", [])):
                        if idx < len(NICHOS_BUY[regimen]):
                            for k, v in a_dict.items(): setattr(NICHOS_BUY[regimen][idx], k, v)
            except Exception: pass
        path_sell = os.path.join(CARPETA_MODELOS_SELL, f"agentes_{regimen}_{VERSION_TAG}.json")
        if os.path.exists(path_sell):
            try:
                with open(path_sell, "r", encoding="utf-8") as f:
                    for idx, a_dict in enumerate(json.load(f).get("agentes", [])):
                        if idx < len(NICHOS_SELL[regimen]):
                            for k, v in a_dict.items(): setattr(NICHOS_SELL[regimen][idx], k, v)
            except Exception: pass
    actualizar_lideres_y_evolucion()

def verificar_y_guardar_progresos():
    global ULTIMO_CONTEO_GUARDADO
    total_acumulado = SISTEMA_TRADES_TOTALES + SHADOW_TRADES_TOTALES
    if total_acumulado - ULTIMO_CONTEO_GUARDADO >= 5:
        actualizar_lideres_y_evolucion()
        guardar_poblacion_agentes()
        ULTIMO_CONTEO_GUARDADO = total_acumulado

def exportar_telemetria_dashboard(regimen, score_buy, score_sell, p_ict_b, p_ses_b, prob_eng_b, p_ict_s, p_ses_s, prob_eng_s, pendiente_ad_h1, permitir_buy, permitir_sell, dir_vela_actual, tick):
    account_info = mt5.account_info()
    data = {
        "timestamp": datetime.datetime.now(TZ_MEXICO).strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION_SISTEMA, "regimen_actual": regimen, "symbol": SYMBOL,
        "precio_bid": tick.bid if tick else 0.0, "precio_ask": tick.ask if tick else 0.0,
        "cuenta": {"balance": account_info.balance if account_info else 0.0, "equity": account_info.equity if account_info else 0.0, "margin_free": account_info.margin_free if account_info else 0.0},
        "candado_ad_h1": {"pendiente": round(pendiente_ad_h1, 4), "permitir_buy": permitir_buy, "permitir_sell": permitir_sell},
        "micro_gatillo_vela": {"direccion_vela_activa": dir_vela_actual},
        "desglose_scores": {
            "buy": {"total": round(score_buy, 4), "xgb_ict": round(p_ict_b, 4), "xgb_sesiones": round(p_ses_b, 4), "engine_ict": round(prob_eng_b, 4)},
            "sell": {"total": round(score_sell, 4), "xgb_ict": round(p_ict_s, 4), "xgb_sesiones": round(p_ses_s, 4), "engine_ict": round(prob_eng_s, 4)}
        },
        "conteo_trades": {"real_trades": SISTEMA_TRADES_TOTALES, "shadow_trades": SHADOW_TRADES_TOTALES},
        "lideres_activos": {"lider_buy_id": LIDERES_BUY[regimen].id, "lider_buy_min_score": round(LIDERES_BUY[regimen].score_minimo, 2), "lider_sell_id": LIDERES_SELL[regimen].id, "lider_sell_min_score": round(LIDERES_SELL[regimen].score_minimo, 2)}
    }
    try:
        with open(ARCHIVO_TELEMETRIA, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)
    except Exception: pass

# ==========================================
# 4. EXTRACCIÓN DE VECTORES DUALES Y EL FIX DE A/D
# ==========================================

def calcular_pendiente_ad(df, periodos=3):
    if df is None or len(df) < periodos + 2: return 0.0
    hl = (df["high"] - df["low"]).replace(0, 1e-10)
    diff = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl * df["tick_volume"]).cumsum().fillna(0).iloc[-2] - (((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl * df["tick_volume"]).cumsum().fillna(0).iloc[-(2 + periodos)]
    return 0.0 if np.isnan(diff) or np.isinf(diff) else float(diff)

def evaluar_direccion_vela_activa(df_m1, tick):
    if df_m1 is None or len(df_m1) < 15 or not tick: return "NEUTRAL"
    open_actual = df_m1["open"].iloc[-1]
    delta_precio = tick.bid - open_actual
    cuerpos_previos = abs(df_m1["close"].iloc[-11:-1] - df_m1["open"].iloc[-11:-1])
    promedio_cuerpo = max(cuerpos_previos.mean(), 1e-5)
    fuerza_expansion = delta_precio / promedio_cuerpo
    if fuerza_expansion >= 0.05: return "ALCISTA"
    elif fuerza_expansion <= -0.05: return "BAJISTA"
    return "NEUTRAL"

def obtener_regimen_actual(df_m1):
    h = datetime.datetime.fromtimestamp(int(df_m1["time"].iloc[-2]), tz=pytz.utc).hour
    if 8 <= h < 13: return "LONDON"
    elif 13 <= h < 17: return "OVERLAP"
    elif 17 <= h < 22: return "NY"
    elif 0 <= h < 9: return "TOKYO"
    else: return "OTROS"

def calculate_williams_r(df, period=14):
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    divisor = (high_max - low_min).replace(0, 1e-10)
    return (((high_max - df["close"]) / divisor) * -100.0).clip(-100.0, 0.0)

def calcular_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def obtener_filling_mode(info):
    if info is None: return mt5.ORDER_FILLING_IOC
    mode = info.filling_mode
    if mode & 1: return mt5.ORDER_FILLING_FOK
    elif mode & 2: return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN

def cargar_modelos():
    ict_buy, ses_buy, ict_sell, ses_sell = None, None, None, None
    try:
        if os.path.exists(MODELO_ICT_BUY_PATH) and os.path.exists(MODELO_SESIONES_BUY_PATH):
            ict_buy, ses_buy = xgb.Booster(), xgb.Booster()
            ict_buy.load_model(MODELO_ICT_BUY_PATH)
            ses_buy.load_model(MODELO_SESIONES_BUY_PATH)
        if os.path.exists(MODELO_ICT_SELL_PATH) and os.path.exists(MODELO_SESIONES_SELL_PATH):
            ict_sell, ses_sell = xgb.Booster(), xgb.Booster()
            ict_sell.load_model(MODELO_ICT_SELL_PATH)
            ses_sell.load_model(MODELO_SESIONES_SELL_PATH)
    except Exception as e: 
        print(f"⚠️ Error al cargar modelos: {e}")
    return (ict_buy, ses_buy), (ict_sell, ses_sell)

def preparar_vectores_cerebros(df_m1, df_m5):
    df_m1["wpr"] = calculate_williams_r(df_m1, 14)
    df_m1["wpr_prev"] = df_m1["wpr"].shift(1)
    
    # =========================================================================
    # [PROPIEDAD INTELECTUAL OCULTA PARA GITHUB]
    # Se omiten las fórmulas matemáticas exactas de los vectores de entrada a 
    # XGBoost para prevenir la ingeniería inversa de los modelos (.json).
    # En producción, aquí se calculan momentum, aceleración, ganchos y ruido.
    # =========================================================================
    df_m1["wpr_macro"] = 0.0          # [REDACTED] Lógica real de WPR Macro
    df_m1["wpr_momentum"] = 0.0       # [REDACTED] Lógica real de Momentum
    df_m1["wpr_aceleracion"] = 0.0    # [REDACTED] Lógica real de Aceleración EWM
    df_m1["wpr_hook_bullish"] = 0.0   # [REDACTED] Detección de gancho alcista
    df_m1["wpr_hook_bearish"] = 0.0   # [REDACTED] Detección de gancho bajista
    df_m1["ratio_ruido_fuerza"] = 0.0 # [REDACTED] Ratio de ruido y fuerza

    df_m5["wpr_5m"] = calculate_williams_r(df_m5, 14)
    wpr_5m_val = df_m5["wpr_5m"].iloc[-2]

    period_liq = 20
    high_max = df_m1["high"].rolling(period_liq).max()
    low_min = df_m1["low"].rolling(period_liq).min()
    rango = (high_max - low_min).replace(0, 1e-10)
    close = df_m1["close"]

    # Variables estructurales (Visibles para demostrar el motor ICT)
    df_m1["dist_bsl"] = high_max - close
    df_m1["dist_ssl"] = close - low_min
    df_m1["pd_ratio"] = (close - low_min) / rango
    df_m1["trend_sintetico"] = np.where(close.ewm(span=20).mean() > close.ewm(span=50).mean(), 1.0, -1.0)

    candle_body_cerrada = abs(df_m1["close"].iloc[-2] - df_m1["open"].iloc[-2])
    avg_body_recent = abs(df_m1["close"] - df_m1["open"]).rolling(14).mean().iloc[-2]

    pd_val = df_m1["pd_ratio"].iloc[-2]
    wpr_t1 = df_m1["wpr"].iloc[-2] 
    wpr_t0 = df_m1["wpr"].iloc[-1] 
    wpr_prev_val = df_m1["wpr_prev"].iloc[-2]

    ict1_buy = (pd_val <= 0.35) or (df_m1["trend_sintetico"].iloc[-2] == 1.0)
    ict2_buy = (wpr_t1 <= -75.0) or (df_m1["dist_ssl"].iloc[-2] < df_m1["dist_bsl"].iloc[-2])
    exiting_bot_buy = (wpr_prev_val <= -80.0) and (wpr_t1 > -80.0)

    prob_buy = ENGINE_ICT_BUY.calculate_probability(ict1_buy, ict2_buy, candle_body_cerrada, avg_body_recent, exiting_bot_buy)

    ict1_sell = (pd_val >= 0.65) or (df_m1["trend_sintetico"].iloc[-2] == -1.0)
    ict2_sell = (wpr_t1 >= -25.0) or (df_m1["dist_bsl"].iloc[-2] < df_m1["dist_ssl"].iloc[-2])
    exiting_top_sell = (wpr_prev_val >= -20.0) and (wpr_t1 < -20.0)

    prob_sell = ENGINE_ICT_SELL.calculate_probability(ict1_sell, ict2_sell, candle_body_cerrada, avg_body_recent, exiting_top_sell)

    feature_names_ict = [
        "wpr_macro", "wpr_prev", "wpr_momentum", "wpr_aceleracion",
        "dist_bsl", "dist_ssl", "pd_ratio", "trend_sintetico",
        "ratio_ruido_fuerza", "wpr_hook_bullish", "wpr_hook_bearish"
    ]
    f_ict = pd.DataFrame([{fn: df_m1[fn].iloc[-2] for fn in feature_names_ict}])

    dt_utc = datetime.datetime.fromtimestamp(int(df_m1["time"].iloc[-2]), tz=pytz.utc)
    h_utc = dt_utc.hour
    dict_sesiones = {fn: df_m1[fn].iloc[-2] for fn in feature_names_ict}
    dict_sesiones.update({
        "is_tokyo": 1.0 if (0 <= h_utc < 9) else 0.0,
        "is_london": 1.0 if (8 <= h_utc < 17) else 0.0,
        "is_ny": 1.0 if (13 <= h_utc < 22) else 0.0,
        "is_overlap": 1.0 if ((8 <= h_utc < 17) and (13 <= h_utc < 22)) else 0.0,
    })
    f_sesiones = pd.DataFrame([dict_sesiones])[feature_names_ict + ["is_tokyo", "is_london", "is_ny", "is_overlap"]]

    rango_vela_actual = max((df_m1["high"].iloc[-2] - df_m1["low"].iloc[-2]), 1e-10)
    mecha_sup = (df_m1["high"].iloc[-2] - max(df_m1["open"].iloc[-2], df_m1["close"].iloc[-2])) / rango_vela_actual
    mecha_inf = (min(df_m1["open"].iloc[-2], df_m1["close"].iloc[-2]) - df_m1["low"].iloc[-2]) / rango_vela_actual
    atr_actual = calcular_atr(df_m1, 14).iloc[-2]

    # EL GANCHO EXACTO (AFUERA -> ADENTRO)
    hook_buy = (wpr_t1 <= -75.0) and (wpr_t0 > wpr_t1)
    hook_sell = (wpr_t1 >= -25.0) and (wpr_t0 < wpr_t1)

    fisica = {
        "wpr": wpr_t1, "wpr_5m": wpr_5m_val, "mecha_sup": mecha_sup, "mecha_inf": mecha_inf, "atr": atr_actual,
        "ict_engine_buy": prob_buy, "ict_engine_sell": prob_sell,
        "hook_buy_valido": hook_buy, "hook_sell_valido": hook_sell
    }
    return f_ict, f_sesiones, fisica

# ==========================================
# 5. EJECUCIÓN Y GESTIÓN DE ÓRDENES DUALES
# ==========================================
def execute_trade_elite(symbol, order_type, price, sl, tp, score_disparo, magic_number, regimen):
    global PICO_MAXIMO_BUY, PICO_MAXIMO_SELL, TIEMPO_ENTRADA_BUY, TIEMPO_ENTRADA_SELL
    info = mt5.symbol_info(symbol)
    if info is None: return False

    # LOTE FIJO ESTRICTO
    lot = LOTE_FIJO
    if lot < info.volume_min: lot = info.volume_min
    
    lider = LIDERES_BUY[regimen] if order_type == mt5.ORDER_TYPE_BUY else LIDERES_SELL[regimen]

    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lot), "type": order_type,
        "price": float(round(price, info.digits)), "sl": float(round(sl, info.digits)), "tp": float(round(tp, info.digits)),
        "deviation": 50, "magic": magic_number, "type_time": mt5.ORDER_TIME_GTC, "type_filling": obtener_filling_mode(info),
    }

    res = mt5.order_send(req)
    if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
        if order_type == mt5.ORDER_TYPE_BUY:
            PICO_MAXIMO_BUY, TIEMPO_ENTRADA_BUY, lbl, tipo_str = 0.0, time.time(), "COMPRA", "BUY"
        else:
            PICO_MAXIMO_SELL, TIEMPO_ENTRADA_SELL, lbl, tipo_str = 0.0, time.time(), "VENTA", "SELL"

        datos_mt5 = {"ticket_mt5": res.deal, "order_mt5": res.order, "agente_id": lider.id, "tipo_op": tipo_str, "volumen": float(lot), "precio_teorico": price, "precio_ejecutado": res.price, "motivo_cierre": "APERTURA_REAL"}
        auditar_calculo(f"DISPARO REAL {lbl} | Order #{res.order} | Vol: {lot}", regimen, score=score_disparo, etiqueta="[DISPARO]", datos_mt5=datos_mt5)
        return True
    return False

def gestionar_cierre_activo(symbol, magic_number, is_buy, tick, fisica, score_direccion, regimen):
    global PICO_MAXIMO_BUY, PICO_MAXIMO_SELL, SISTEMA_TRADES_TOTALES
    posiciones = mt5.positions_get(symbol=symbol, magic=magic_number)
    if not posiciones or len(posiciones) == 0:
        if is_buy: PICO_MAXIMO_BUY = 0.0
        else: PICO_MAXIMO_SELL = 0.0
        return False

    posicion = posiciones[0]
    lider = LIDERES_BUY[regimen] if is_buy else LIDERES_SELL[regimen]
    info = mt5.symbol_info(symbol)
    point = info.point if info.point > 0 else 0.01

    if is_buy:
        beneficio_usd = tick.bid - posicion.price_open
        pico_actual = PICO_MAXIMO_BUY
        tiempo_ent = TIEMPO_ENTRADA_BUY
    else:
        beneficio_usd = posicion.price_open - tick.ask
        pico_actual = PICO_MAXIMO_SELL
        tiempo_ent = TIEMPO_ENTRADA_SELL

    salida_motivo = None

    # ========================================================
    # 1. EL FUSIBLE DE EMERGENCIA (PIDE -$0.45 ESTRICTO)
    # ========================================================
    limite_emergencia = 45.0 * point
    if beneficio_usd <= -limite_emergencia:
        salida_motivo = "Fusible Emergencia (-$0.45)"

    # ========================================================
    # 2. SI NO HAY PELIGRO, RESPIRA SUS 90 SEGUNDOS
    # ========================================================
    elif (time.time() - tiempo_ent) < TIEMPO_RESPIRO_SEGUNDOS and beneficio_usd < (fisica["atr"] * FACTOR_COLCHON_ATR):
        return True

    if beneficio_usd > pico_actual:
        if is_buy: PICO_MAXIMO_BUY = beneficio_usd
        else: PICO_MAXIMO_SELL = beneficio_usd
        pico_actual = beneficio_usd

    wpr_eval = fisica["wpr_5m"] if lider.usar_wpr_5m else fisica["wpr"]
    
    # 3. GESTIÓN ORIGINAL DE CIERRES V5
    if not salida_motivo:
        if beneficio_usd > 0:
            if beneficio_usd > (fisica["atr"] * FACTOR_COLCHON_ATR) and beneficio_usd <= (pico_actual * (1.0 - lider.retroceso_pico_exit)):
                salida_motivo = f"Retroceso de Pico (${pico_actual:.2f} -> ${beneficio_usd:.2f} USD)"
            elif is_buy and (wpr_eval >= lider.wpr_exit or score_direccion < lider.reversal_score_limit):
                salida_motivo = "Agotamiento WPR/Score Compra"
            elif (not is_buy) and (wpr_eval <= lider.wpr_exit or score_direccion < lider.reversal_score_limit):
                salida_motivo = "Agotamiento WPR/Score Venta"
            elif is_buy and fisica["mecha_sup"] >= lider.ratio_mecha_exit:
                salida_motivo = "Rechazo por Mecha Superior"
            elif (not is_buy) and fisica["mecha_inf"] >= lider.ratio_mecha_exit:
                salida_motivo = "Rechazo por Mecha Inferior"

    if salida_motivo:
        tipo_cierre = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        precio_cierre = tick.bid if is_buy else tick.ask

        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": posicion.volume, "type": tipo_cierre,
            "position": posicion.ticket, "price": precio_cierre, "deviation": 50, "magic": magic_number,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": obtener_filling_mode(mt5.symbol_info(posicion.symbol)),
        }
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            SISTEMA_TRADES_TOTALES += 1
            datos_mt5 = {"ticket_mt5": res.deal, "order_mt5": posicion.ticket, "profit_neto": round(posicion.profit, 2), "motivo_cierre": salida_motivo}
            auditar_calculo(f"CIERRE MT5 [{salida_motivo}] | Profit: ${posicion.profit:.2f}", regimen, etiqueta="[CIERRE]", datos_mt5=datos_mt5)
            verificar_y_guardar_progresos()
    return True

# ==========================================
# 6. BUCLE PRINCIPAL (ORIGINAL V5)
# ==========================================
def main():
    global SHADOW_TRADES_TOTALES, SISTEMA_TRADES_TOTALES
    if not mt5.initialize(): quit()

    (ict_buy, ses_buy), (ict_sell, ses_sell) = cargar_modelos()
    cargar_poblacion_agentes()
    ultima_vela_operada = None

    auditar_calculo(f"🚀 Sistema Inicializado [{VERSION_SISTEMA}] | V5 ORIGINAL + HOOK AFUERA->ADENTRO", "SISTEMA", etiqueta="[SISTEMA]")

    while True:
        time.sleep(1)
        if not mt5.terminal_info(): mt5.initialize(); continue

        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick: continue

        df_1m = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 60))
        df_5m = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 60))
        df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 60))

        if len(df_1m) < 50 or len(df_5m) < 20 or len(df_h1) < 10: continue

        # FIX A/D: Ahora usa 3 periodos para que NUNCA vuelva a dar 0.00
        pendiente_ad_h1 = calcular_pendiente_ad(df_h1, 3)
        if pendiente_ad_h1 == 0.0: pendiente_ad_h1 = calcular_pendiente_ad(df_5m, 3)

        permitir_buy = bool(pendiente_ad_h1 > 0)
        permitir_sell = bool(pendiente_ad_h1 < 0)

        dir_vela_actual = evaluar_direccion_vela_activa(df_1m, tick)
        regimen_actual = obtener_regimen_actual(df_1m)
        f_ict, f_sesiones, fisica = preparar_vectores_cerebros(df_1m, df_5m)

        p_ict_b = float(ict_buy.predict(xgb.DMatrix(f_ict)).flat[0]) if ict_buy is not None else 0.5
        p_ses_b = float(ses_buy.predict(xgb.DMatrix(f_sesiones)).flat[0]) if ses_buy is not None else 0.5
        p_ict_s = float(ict_sell.predict(xgb.DMatrix(f_ict)).flat[0]) if ict_sell is not None else 0.5
        p_ses_s = float(ses_sell.predict(xgb.DMatrix(f_sesiones)).flat[0]) if ses_sell is not None else 0.5

        score_buy = ((0.35 * p_ict_b) + (0.35 * p_ses_b) + (0.40 * fisica["ict_engine_buy"]["probability"]))
        score_sell = ((0.35 * p_ict_s) + (0.35 * p_ses_s) + (0.40 * fisica["ict_engine_sell"]["probability"]))

        exportar_telemetria_dashboard(regimen_actual, score_buy, score_sell, p_ict_b, p_ses_b, fisica["ict_engine_buy"]["probability"], p_ict_s, p_ses_s, fisica["ict_engine_sell"]["probability"], pendiente_ad_h1, permitir_buy, permitir_sell, dir_vela_actual, tick)

        gestionar_cierre_activo(SYMBOL, MAGIC_BUY, True, tick, fisica, score_buy, regimen_actual)
        gestionar_cierre_activo(SYMBOL, MAGIC_SELL, False, tick, fisica, score_sell, regimen_actual)

        # EVALUACIÓN SHADOW TRADING RESTAURADA
        precio_buy = tick.ask
        precio_sell = tick.bid
        atr_actual = fisica["atr"]
        shadow_trades_cerrados_iteracion = 0

        for agente in NICHOS_BUY[regimen_actual]:
            if agente.evaluar_shadow_trade(precio_buy, score_buy, fisica, atr_actual, permiso_entrada=permitir_buy): shadow_trades_cerrados_iteracion += 1
        for agente in NICHOS_SELL[regimen_actual]:
            if agente.evaluar_shadow_trade(precio_sell, score_sell, fisica, atr_actual, permiso_entrada=permitir_sell): shadow_trades_cerrados_iteracion += 1

        if shadow_trades_cerrados_iteracion > 0:
            SHADOW_TRADES_TOTALES += shadow_trades_cerrados_iteracion
            verificar_y_guardar_progresos()

        tiempo_actual = int(df_1m["time"].iloc[-1])
        if tiempo_actual == ultima_vela_operada: continue
        
        # ==========================================
        # ENTRADAS REALES (CON EL HOOK AFUERA->ADENTRO)
        # ==========================================
        if permitir_buy and dir_vela_actual == "ALCISTA" and fisica["hook_buy_valido"]:
            pos_buy = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_BUY)
            if (not pos_buy) and fisica["ict_engine_buy"]["triggered"]:
                lider_b = LIDERES_BUY[regimen_actual]
                if (score_buy + 1e-5) >= lider_b.score_minimo:
                    exito = execute_trade_elite(SYMBOL, mt5.ORDER_TYPE_BUY, tick.ask, tick.bid - (2 * atr_actual), tick.bid + (6 * atr_actual), score_buy, MAGIC_BUY, regimen_actual)
                    if exito: ultima_vela_operada = tiempo_actual

        if permitir_sell and dir_vela_actual == "BAJISTA" and fisica["hook_sell_valido"]:
            pos_sell = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_SELL)
            if (not pos_sell) and fisica["ict_engine_sell"]["triggered"]:
                lider_s = LIDERES_SELL[regimen_actual]
                if (score_sell + 1e-5) >= lider_s.score_minimo:
                    exito = execute_trade_elite(SYMBOL, mt5.ORDER_TYPE_SELL, tick.bid, tick.ask + (2 * atr_actual), tick.ask - (6 * atr_actual), score_sell, MAGIC_SELL, regimen_actual)
                    if exito: ultima_vela_operada = tiempo_actual

if __name__ == "__main__":
    main()
