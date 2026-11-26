#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor online del port de Demon's Souls (KytyPS5), entrega online2.

Para que sirve: hace de servidor de juego para los jugadores del port (lo que antes hacian los servidores de
Sony/Bluepoint). Con KYTY_ONLINE=1 el emulador manda aqui todas las peticiones HTTP del juego.

Que hace (formato sacado del eboot 01.007.000 con Ghidra; ver INFORME.md):
  sobre de respuesta: {"ServerState":0,"data":...}            (DSNetworkClient::_AnalyzeMessage @0x6605b0)
  /dsmaster/ss        getServerState   datos del servidor      (@0x6614a0)
  /dsgame/playerdata  login, addNewAccount, initializeMultiplay, getMultiplayGrade, getTimeMessage
  /dsgame/bloodmessages addBloodMessage, getBloodMessage (por zona), updateBloodMessageGrade (valorar),
                      deleteBloodMessage, getBloodMessageGrade
  /dsgame/replay      addReplayData (manchas de sangre), getReplayList (por zona), getReplayData (descargar)
  /dsgame/character   setWanderingGhost / getWanderingGhost (fantasmas que deambulan)
  /dsgame/qwc         getQWCData (tendencia del mundo)
  /dsgame/sos         addSummonSign, deleteSummonSign, getSummonSigns, getForceSummons (sin P2P: no se listan
                      senales de otros salvo con --mostrar-senales, porque invocar necesita la fase C)
  cualquier otra cosa: se apunta y se contesta {"ServerState":0,"data":{}} (respuesta vacia valida)

Datos: SQLite en registro/online.db; todas las peticiones en registro/peticiones.jsonl.
Uso:   python servidor.py [puerto] [--mostrar-senales] [--ver-propios]
       Escucha en 0.0.0.0 (los demas PCs ponen KYTY_ONLINE_SERVER=<IP de este PC>:<puerto>).
Solo biblioteca estandar de Python 3.
"""
import json
import os
import random
import sqlite3
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
PUERTO = int(ARGS[0]) if ARGS else 8090
MOSTRAR_SENALES = "--mostrar-senales" in sys.argv
VER_PROPIOS = "--ver-propios" in sys.argv  # ver tambien tus propios mensajes/manchas (para probar con un solo PC)
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registro")
os.makedirs(DIR, exist_ok=True)
DB = os.path.join(DIR, "online.db")
LOCK = threading.Lock()

# Limites que el juego aplica a los valores de getServerState (@0x6614a0): intervalos 20..48, listas 50..100.
ESTADO = {
    "serverState": 0,                       # tiene que ser un numero: si no, el juego no lee el resto
    "serverStateMsg": "",
    "serverSandbox": "kyty",
    "gameServerStatusMessageInterval": 600,
    "enableOnlineMultiplay": False,         # coop/invasion (fase C) apagado
    "enableWanderingGhost": True,
    "getWanderingGhostInterval": 30,
    "setWanderingGhostInterval": 30,
    "getBloodMessageNum": 50,
    "getReplayListNum": 50,
    "getBluePhantomNum": 0,
    "getRedPhantomNum": 0,
    "getBlackPhantomNum": 0,
    "getOrangePhantomNum": 0,
    "getSOSQueryInternvalMin": 30,
    "getSOSRefreshInternval": 30,
    "getSOSQueryInternvalMax": 60,
}


def db():
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def crea_tablas():
    with LOCK, db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS cuentas (account TEXT PRIMARY KEY, creada REAL, ultimo REAL, grade_pendiente INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS mensajes (id TEXT PRIMARY KEY, account TEXT, block INTEGER, px REAL, py REAL, pz REAL,
            ax REAL, ay REAL, az REAL, message_id INTEGER, main_msg_id INTEGER, add_msg_cate_id INTEGER,
            grade_up INTEGER DEFAULT 0, grade_down INTEGER DEFAULT 0, creado REAL, extra TEXT);
        CREATE TABLE IF NOT EXISTS replays (id TEXT PRIMARY KEY, account TEXT, block INTEGER, px REAL, py REAL, pz REAL,
            ax REAL, ay REAL, az REAL, datos TEXT, creado REAL, extra TEXT);
        CREATE TABLE IF NOT EXISTS fantasmas (account TEXT, block INTEGER, datos TEXT, creado REAL, PRIMARY KEY(account, block));
        CREATE TABLE IF NOT EXISTS senales (account TEXT PRIMARY KEY, block INTEGER, sign_type INTEGER, datos TEXT, creado REAL);
        """)


def apunta(fichero, obj):
    with LOCK:
        with open(os.path.join(DIR, fichero), "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# --- lectura tolerante de campos -------------------------------------------------------------------------
def entero(v, defecto=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return defecto


def real(v, defecto=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return defecto


def vec(v):
    """pos/ang: objeto {x,y,z}, lista [x,y,z] o un numero (solo y)."""
    if isinstance(v, dict):
        return real(v.get("x")), real(v.get("y")), real(v.get("z"))
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return real(v[0]), real(v[1]), real(v[2])
    if isinstance(v, (int, float)):
        return 0.0, float(v), 0.0
    return 0.0, 0.0, 0.0


def bloque(p):
    for k in ("blockId", "block_id", "blockID"):
        if k in p:
            return entero(p[k])
    return 0


def cuenta(p):
    return str(p.get("AccountId") or p.get("accountId") or "0")


def uid():
    return str(uuid.uuid4()).upper()


def posiciones(fila):
    return {"pos_x": fila["px"], "pos_y": fila["py"], "pos_z": fila["pz"],
            "ang_x": fila["ax"], "ang_y": fila["ay"], "ang_z": fila["az"]}


def filtro_propio(acc):
    return ("", ()) if VER_PROPIOS else (" AND account <> ?", (acc,))


# --- comandos ----------------------------------------------------------------------------------------------
def c_getServerState(p, h):
    d = dict(ESTADO)
    d["gameServerUrl"] = "http://" + h.headers.get("Host", "127.0.0.1:%d" % PUERTO)
    return d


def c_login(p, h):
    acc = cuenta(p)
    with LOCK, db() as c:
        c.execute("INSERT INTO cuentas(account, creada, ultimo) VALUES(?,?,?) ON CONFLICT(account) DO UPDATE SET ultimo=?",
                  (acc, time.time(), time.time(), time.time()))
    return {"loginState": 1}            # 1 = dentro (0 pediria aceptar "agreement")


def c_addNewAccount(p, h):
    acc = cuenta(p)
    with LOCK, db() as c:
        nueva = c.execute("SELECT 1 FROM cuentas WHERE account=?", (acc,)).fetchone() is None
        c.execute("INSERT OR IGNORE INTO cuentas(account, creada, ultimo) VALUES(?,?,?)", (acc, time.time(), time.time()))
    return {"createdNewAccount": nueva}


def c_vacio(p, h):
    return {}


def c_getMultiplayGrade(p, h):
    return {"multiPlayCount": 0}


def c_getTimeMessage(p, h):
    return {"timeMessages": [], "timeMessageIds": []}


def c_getQWCData(p, h):
    # data es una lista de 7 objetos {wb, lr} (@0x672e40). Sin datos de tendencia global: neutro.
    return [{"wb": 0, "lr": 0} for _ in range(7)]


def c_addBloodMessage(p, h):
    px, py, pz = vec(p.get("pos"))
    ax, ay, az = vec(p.get("ang"))
    extra = {k: v for k, v in p.items() if k not in ("cmd", "pos", "ang")}
    with LOCK, db() as c:
        c.execute("INSERT INTO mensajes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0,0,?,?)",
                  (uid(), cuenta(p), bloque(p), px, py, pz, ax, ay, az, entero(p.get("messageId")),
                   entero(p.get("mainMsgId")), entero(p.get("addMsgCateId")), time.time(), json.dumps(extra)))
    return {}


def c_getBloodMessage(p, h):
    num = entero(p.get("messageNum"), ESTADO["getBloodMessageNum"]) or ESTADO["getBloodMessageNum"]
    num = max(1, min(num, 100))
    sql, args = filtro_propio(cuenta(p))
    with LOCK, db() as c:
        filas = c.execute("SELECT * FROM mensajes WHERE block=?" + sql + " ORDER BY creado DESC LIMIT 500",
                          (bloque(p),) + args).fetchall()
    filas = random.sample(filas, min(num, len(filas)))
    lista = []
    for f in filas:
        m = {"blood_msg_id": f["id"], "account_id": f["account"], "block_id": f["block"]}
        m.update(posiciones(f))
        m.update({"message_id": f["message_id"], "main_msg_id": f["main_msg_id"],
                  "add_msg_cate_id": f["add_msg_cate_id"], "grade_up": f["grade_up"], "grade_down": f["grade_down"]})
        lista.append(m)
    return {"bloodMsgs": lista}


def c_updateBloodMessageGrade(p, h):
    mid = str(p.get("bloodMsgId") or "").upper()
    up, down = entero(p.get("upgrade")), entero(p.get("downgrade"))
    with LOCK, db() as c:
        f = c.execute("SELECT account FROM mensajes WHERE upper(id)=?", (mid,)).fetchone()
        if f is not None:
            c.execute("UPDATE mensajes SET grade_up=grade_up+?, grade_down=grade_down+? WHERE upper(id)=?", (up, down, mid))
            if up > 0:
                c.execute("UPDATE cuentas SET grade_pendiente=grade_pendiente+? WHERE account=?", (up, f["account"]))
    return {}


def c_getBloodMessageGrade(p, h):
    # Valoraciones nuevas de mis mensajes desde la ultima vez (el juego cura al jugador por ellas).
    acc = cuenta(p)
    with LOCK, db() as c:
        f = c.execute("SELECT grade_pendiente FROM cuentas WHERE account=?", (acc,)).fetchone()
        n = f["grade_pendiente"] if f else 0
        c.execute("UPDATE cuentas SET grade_pendiente=0 WHERE account=?", (acc,))
    return {"blood_message_grade": n}


def c_deleteBloodMessage(p, h):
    with LOCK, db() as c:
        c.execute("DELETE FROM mensajes WHERE upper(id)=? AND account=?", (str(p.get("bloodMsgId") or "").upper(), cuenta(p)))
    return {}


def c_addReplayData(p, h):
    px, py, pz = vec(p.get("pos"))
    ax, ay, az = vec(p.get("ang"))
    datos = p.get("replayData")
    if not isinstance(datos, str):
        datos = json.dumps(datos)
    extra = {k: v for k, v in p.items() if k not in ("cmd", "pos", "ang", "replayData")}
    with LOCK, db() as c:
        c.execute("INSERT INTO replays VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                  (uid(), cuenta(p), bloque(p), px, py, pz, ax, ay, az, datos, time.time(), json.dumps(extra)))
    return {}


def c_getReplayList(p, h):
    num = entero(p.get("replayNum") or p.get("maxReplayNum"), ESTADO["getReplayListNum"]) or ESTADO["getReplayListNum"]
    num = max(1, min(num, 100))
    sql, args = filtro_propio(cuenta(p))
    with LOCK, db() as c:
        filas = c.execute("SELECT * FROM replays WHERE block=?" + sql + " ORDER BY creado DESC LIMIT 500",
                          (bloque(p),) + args).fetchall()
    filas = random.sample(filas, min(num, len(filas)))
    lista = []
    for f in filas:
        r = {"replay_id": f["id"], "account_id": f["account"], "block_id": f["block"]}
        r.update(posiciones(f))
        lista.append(r)
    return {"replays": lista}


def c_getReplayData(p, h):
    rid = str(p.get("replayMsgId") or "").upper()
    with LOCK, db() as c:
        f = c.execute("SELECT id, datos FROM replays WHERE upper(id)=?", (rid,)).fetchone()
    if f is None:
        return {}
    return {"replayData": f["datos"], "replayMsgId": f["id"]}


def c_setWanderingGhost(p, h):
    datos = p.get("replayData")
    if not isinstance(datos, str) or not datos:
        return {}
    with LOCK, db() as c:
        c.execute("INSERT OR REPLACE INTO fantasmas VALUES(?,?,?,?)", (cuenta(p), bloque(p), datos, time.time()))
    return {}


def c_getWanderingGhost(p, h):
    num = max(0, min(entero(p.get("maxGhostNum"), 10), 50))
    sql, args = filtro_propio(cuenta(p))
    limite = time.time() - 30 * 60          # fantasmas de la ultima media hora
    with LOCK, db() as c:
        filas = c.execute("SELECT datos FROM fantasmas WHERE block=? AND creado>?" + sql + " ORDER BY creado DESC LIMIT ?",
                          (bloque(p), limite) + args + (num,)).fetchall()
    return {"sosWandering": [], "wandering": [f["datos"] for f in filas]}


def c_addSummonSign(p, h):
    # El juego espera su propia senal de vuelta en newSigns.signList[0] (accountId + pos), @0x6605b0 caso 6.
    sign_type = entero(p.get("signType"))
    pos = p.get("pos") if isinstance(p.get("pos"), dict) else dict(zip("xyz", vec(p.get("pos"))))
    senal = {"accountId": cuenta(p), "platform": entero(p.get("platform"), 1), "pos": pos}
    if "playerData" in p:
        senal["playerData"] = p["playerData"]
    with LOCK, db() as c:
        c.execute("INSERT OR REPLACE INTO senales VALUES(?,?,?,?,?)",
                  (cuenta(p), bloque(p), sign_type, json.dumps(senal), time.time()))
    return {"newSigns": {"signType": sign_type, "signList": [senal]}}


def c_deleteSummonSign(p, h):
    with LOCK, db() as c:
        c.execute("DELETE FROM senales WHERE account=?", (cuenta(p),))
    return {}


def c_getSummonSigns(p, h):
    if not MOSTRAR_SENALES:
        return {"newSigns": []}
    acc = cuenta(p)
    with LOCK, db() as c:
        filas = c.execute("SELECT sign_type, datos FROM senales WHERE account<>? AND creado>? ORDER BY creado DESC LIMIT 20",
                          (acc, time.time() - 600)).fetchall()
    grupos = {}
    for f in filas:
        grupos.setdefault(f["sign_type"], []).append(json.loads(f["datos"]))
    return {"newSigns": [{"signType": t, "ignoredPassword": 0, "signList": l} for t, l in grupos.items()]}


def c_getForceSummons(p, h):
    return {"newSigns": []}


COMANDOS = {
    "getServerState": c_getServerState, "login": c_login, "addNewAccount": c_addNewAccount,
    "initializeMultiplay": c_vacio, "initializeCharacter": c_vacio, "getMultiplayGrade": c_getMultiplayGrade,
    "getTimeMessage": c_getTimeMessage, "getQWCData": c_getQWCData,
    "addBloodMessage": c_addBloodMessage, "getBloodMessage": c_getBloodMessage,
    "updateBloodMessageGrade": c_updateBloodMessageGrade, "getBloodMessageGrade": c_getBloodMessageGrade,
    "deleteBloodMessage": c_deleteBloodMessage,
    "addReplayData": c_addReplayData, "getReplayList": c_getReplayList, "getReplayData": c_getReplayData,
    "setWanderingGhost": c_setWanderingGhost, "getWanderingGhost": c_getWanderingGhost,
    "addSummonSign": c_addSummonSign, "deleteSummonSign": c_deleteSummonSign,
    "getSummonSigns": c_getSummonSigns, "getForceSummons": c_getForceSummons,
}


class Manejador(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _responde(self, obj):
        cuerpo = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _atiende(self):
        largo = int(self.headers.get("Content-Length") or 0)
        texto = (self.rfile.read(largo) if largo else b"").decode("utf-8", "replace")
        try:
            p = json.loads(texto) if texto.strip() else {}
        except ValueError:
            p = {"_no_json": texto[:4000]}
        if not isinstance(p, dict):
            p = {"_cuerpo": p}
        ruta = self.path.split("?")[0]
        cmd = str(p.get("cmd", "")) or ("getServerState" if ruta.endswith("/dsmaster/ss") else "")
        funcion = COMANDOS.get(cmd)
        try:
            data = funcion(p, self) if funcion else {}
            nota = "ok" if funcion else "desconocido (respuesta vacia)"
        except Exception as e:  # nunca dejar al juego sin respuesta valida
            data, nota = {}, "ERROR %r" % e
        resumen = {k: (v if len(json.dumps(v)) < 300 else "<%d bytes>" % len(json.dumps(v))) for k, v in p.items()}
        apunta("peticiones.jsonl", {"hora": time.strftime("%Y-%m-%d %H:%M:%S"), "cliente": self.client_address[0],
                                    "metodo": self.command, "ruta": self.path,
                                    "url_original": self.headers.get("X-Kyty-Url", ""), "cmd": cmd,
                                    "peticion": resumen, "nota": nota})
        print("%s %-15s %-6s %-22s %s" % (time.strftime("%H:%M:%S"), self.client_address[0], self.command,
                                          cmd or ruta, nota), flush=True)
        self._responde({"ServerState": 0, "data": data})

    do_GET = _atiende
    do_POST = _atiende
    do_PUT = _atiende
    do_DELETE = _atiende
    do_PATCH = _atiende


if __name__ == "__main__":
    crea_tablas()
    print("Servidor online de Demon's Souls en el puerto %d (todas las interfaces)." % PUERTO, flush=True)
    print("Datos y registro en %s. Deja esta ventana abierta. Ctrl+C para cerrar." % DIR, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PUERTO), Manejador).serve_forever()
