"""
DQN para el mismo ejercicio que Q-Learning y SARSA (trader_rl.py).

Qué se mantiene igual, para que los números se puedan comparar:
  * El entorno es SimpleTradingEnv, importado, no reescrito.
  * Cada episodio usa una serie nueva del mismo generador, n_steps = 500.
  * Validación: 50 series, semilla 2024. Test: 100 series, semilla 7777.
  * Durante el entrenamiento se evalúa la política greedy (ε = 0) cada 100
    episodios sobre las 10 primeras series de validación.
  * Métrica: log-retorno medio log(B_T / B_0), más mediana del balance y
    número de operaciones. Baselines: aleatorio y buy & hold.
  * Máscara de acciones válidas (sin activo no se vende; con activo no se
    compra) y desempate aleatorio. Misma exploración ε que en la tabla:
    decaimiento exponencial calibrado por episodio.
  * γ descuenta la misma recompensa de un paso.

Qué cambia, porque es DQN y no una tabla:
  * El estado es la observación continua (tendencia, posición). No hay bins.
  * Q es una red pequeña. El update usa replay y una red objetivo (DQN de Mnih
    et al., 2015), no el backup tabular.
  * El análogo de α es la tasa de aprendizaje de Adam.
  * La tendencia se divide por su percentil 99 de calibración. Si no, la posición
    (0 o 1) aplasta a la tendencia (±0.06 en tres generadores) y la red no la usa.
    Es el mismo papel que los cuantiles en la tabla: repartir el rango que importa.
"""
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from trader_rl import (COMPRAR, MANTENER, VENDER, SimpleTradingEnv, calcular_tendencias,
                       evaluar, generar_series, politica_aleatoria, politica_buy_and_hold)

# Una red de este tamaño no gana nada con varios hilos; el coste es el arranque.
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass


def escala_tendencia(generador, n_series=100, n_steps=500, seed=12345):
    """Escala para dejar la tendencia en torno a [-1, 1].

    En tres de los cuatro generadores la tendencia cabe en ±0.06, y la posición
    es 0 o 1. Sin esta escala la red ignora la tendencia. Los cortes salen de
    series de calibración (la misma semilla que Discretizador), no del test.
    """
    series = generar_series(generador, n_series, n_steps, seed)
    tendencias = np.concatenate([calcular_tendencias(p) for p in series])
    return float(np.quantile(np.abs(tendencias), 0.99))

# Sin activo, vender = mantener. Con activo, comprar = mantener.
ACCIONES_VALIDAS = (np.array([MANTENER, COMPRAR]), np.array([VENDER, MANTENER]))


@dataclass
class ConfigDQN:
    """Hiperparámetros. Los de exploración y el horizonte replican Config de trader_rl."""
    gamma: float = 0.95
    lr: float = 1e-4          # 1e-3 llega a olvidar la política; el grid igual lo compara
    eps_inicial: float = 1.0
    eps_final: float = 0.01
    frac_exploracion: float = 0.8
    n_episodios: int = 5000
    n_steps: int = 500
    batch: int = 64
    buffer: int = 50_000
    calentamiento: int = 2_000   # transiciones antes del primer update
    actualiza_cada: int = 8      # pasos de entorno entre updates (DQN clásico)
    target_cada: int = 500       # pasos de entorno entre copias de la red objetivo
    hidden: int = 128
    grad_clip: float = 1.0

    def epsilon(self, episodio):
        """Misma fórmula que Config.epsilon: ε llega a eps_final en frac_exploracion · N."""
        d = (self.eps_final / self.eps_inicial) ** (1 / (self.frac_exploracion * self.n_episodios))
        return max(self.eps_final, self.eps_inicial * d ** episodio)


class RedQ(nn.Module):
    """Q(s) para las 3 acciones. La última capa nace a cero, como la tabla Q."""

    def __init__(self, hidden):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.red[-1].weight)
        nn.init.zeros_(self.red[-1].bias)

    def forward(self, x):
        return self.red(x)


class Buffer:
    """Replay de tamaño fijo. El muestreo usa un RNG propio, no el de los precios."""

    def __init__(self, capacidad, rng):
        self.s = np.zeros((capacidad, 2), np.float32)
        self.a = np.zeros(capacidad, np.int64)
        self.r = np.zeros(capacidad, np.float32)
        self.s2 = np.zeros((capacidad, 2), np.float32)
        self.fin = np.zeros(capacidad, np.float32)
        self.capacidad = capacidad
        self.rng = rng
        self.n = 0
        self.i = 0

    def push(self, s, a, r, s2, fin):
        self.s[self.i] = s
        self.a[self.i] = a
        self.r[self.i] = r
        self.s2[self.i] = s2
        self.fin[self.i] = fin
        self.i = (self.i + 1) % self.capacidad
        self.n = min(self.n + 1, self.capacidad)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.fin[idx]


def _perdida_dqn(q, q_fija, st, s2t, at, rt, ft, gamma):
    """Misma pérdida que el update anterior. El objetivo no lleva gradiente."""
    q_sa = q(st).gather(1, at).squeeze(1)
    with torch.no_grad():
        q_sig = q_fija(s2t).clone()
        q_sig[s2t[:, 1] == 0, VENDER] = -1e9
        q_sig[s2t[:, 1] == 1, COMPRAR] = -1e9
        objetivo = rt + gamma * q_sig.max(dim=1).values * (1.0 - ft)
    return F.smooth_l1_loss(q_sa, objetivo)


_PASO_DQN = None


def paso_dqn(*args):
    """Update compilado. Si la compilación falla, se usa la función tal cual."""
    global _PASO_DQN
    if _PASO_DQN is None:
        try:
            _PASO_DQN = torch.compile(_perdida_dqn)
        except Exception:
            _PASO_DQN = _perdida_dqn
    try:
        return _PASO_DQN(*args)
    except Exception:
        _PASO_DQN = _perdida_dqn
        return _PASO_DQN(*args)


class AgenteDQN:
    def __init__(self, cfg, seed, escala=1.0):
        self.cfg = cfg
        self.escala = float(escala)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.q = RedQ(cfg.hidden)
        self.q_fija = RedQ(cfg.hidden)
        self.q_fija.load_state_dict(self.q.state_dict())
        for p in self.q_fija.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = Buffer(cfg.buffer, self.rng)
        self._pasos = 0
        h = cfg.hidden
        b = cfg.batch
        # Pesos de las 3 capas lineales, copiados tras cada paso de Adam.
        # La acción usa estos arrays: el forward de un solo estado en PyTorch
        # costaba más que la multiplicación.
        self._lineales = (self.q.red[0], self.q.red[2], self.q.red[4])
        self._W = (np.empty((h, 2), np.float32), np.empty((h, h), np.float32), np.empty((3, h), np.float32))
        self._b = (np.empty(h, np.float32), np.empty(h, np.float32), np.empty(3, np.float32))
        self._ent = np.empty(2, np.float32)
        self._hid = np.empty(h, np.float32)
        self._hid2 = np.empty(h, np.float32)
        self._qout = np.empty(3, np.float32)
        self.volcar_pesos()
        # Batch fijo, compartido con numpy, para no reservar tensores en cada update.
        self._s_np = np.empty((b, 2), np.float32)
        self._a_np = np.empty((b, 1), np.int64)
        self._r_np = np.empty(b, np.float32)
        self._s2_np = np.empty((b, 2), np.float32)
        self._f_np = np.empty(b, np.float32)
        self._s_t = torch.from_numpy(self._s_np)
        self._a_t = torch.from_numpy(self._a_np)
        self._r_t = torch.from_numpy(self._r_np)
        self._s2_t = torch.from_numpy(self._s2_np)
        self._f_t = torch.from_numpy(self._f_np)

    def volcar_pesos(self):
        """Copia los pesos actuales de Q a los arrays con los que se elige la acción."""
        for w, b, capa in zip(self._W, self._b, self._lineales):
            np.copyto(w, capa.weight.detach().numpy())
            np.copyto(b, capa.bias.detach().numpy())

    def codificar(self, obs):
        """Observación del entorno -> entrada de la red, con la tendencia escalada."""
        return np.array([float(obs[0]) / self.escala, float(obs[1])], np.float32)

    def valores(self, obs):
        ent = self._ent
        ent[0] = float(obs[0]) / self.escala
        ent[1] = float(obs[1])
        np.dot(self._W[0], ent, out=self._hid)
        self._hid += self._b[0]
        np.maximum(self._hid, 0, out=self._hid)
        np.dot(self._W[1], self._hid, out=self._hid2)
        self._hid2 += self._b[1]
        np.maximum(self._hid2, 0, out=self._hid2)
        np.dot(self._W[2], self._hid2, out=self._qout)
        self._qout += self._b[2]
        return self._qout.copy()

    def accion(self, obs, epsilon):
        """ε-greedy sobre las acciones válidas, con desempate aleatorio."""
        validas = ACCIONES_VALIDAS[int(obs[1])]
        if self.rng.random() < epsilon:
            return int(validas[self.rng.integers(2)])
        q = self.valores(obs)[validas]
        if q[0] == q[1]:
            return int(validas[self.rng.integers(2)])
        return int(validas[int(q.argmax())])

    def politica_greedy(self):
        return lambda obs: self.accion(obs, 0.0)

    def aprender(self):
        """Un paso de gradiente. Devuelve la pérdida, o None si aún no hay batch."""
        cfg = self.cfg
        if self.buffer.n < max(cfg.batch, cfg.calentamiento):
            return None
        s, a, r, s2, fin = self.buffer.sample(cfg.batch)
        self._s_np[:] = s
        self._a_np[:, 0] = a
        self._r_np[:] = r
        self._s2_np[:] = s2
        self._f_np[:] = fin
        loss = paso_dqn(self.q, self.q_fija, self._s_t, self._s2_t, self._a_t,
                        self._r_t, self._f_t, cfg.gamma)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.opt.step()
        self.volcar_pesos()
        return float(loss.detach())


def entrenar(cfg, generador, seed=0, series_validacion=None, eval_cada=100, al_evaluar=None,
             progreso=None, progreso_cada_s=60):
    """Entrena un DQN. Cada episodio es una serie nueva de `generador`.

    `al_evaluar(episodio, hist, agente)` se llama tras cada evaluación greedy,
    con `hist` todavía en listas. Sirve para redibujar el notebook.
    Si `progreso` es un texto, se imprime más o menos cada `progreso_cada_s`
    segundos para ver que el proceso sigue vivo.
    """
    escala = escala_tendencia(generador, n_steps=cfg.n_steps)
    np.random.seed(seed)  # los generadores usan el RNG global; el agente no
    agente = AgenteDQN(cfg, seed, escala)
    hist = {"recompensa": [], "balance": [], "operaciones": [], "perdida": [], "epsilon": [],
            "eval_episodio": [], "eval_logret": []}
    ultimo_aviso = time.perf_counter()

    for ep in range(cfg.n_episodios):
        eps = cfg.epsilon(ep)
        env = SimpleTradingEnv(generador(cfg.n_steps))
        obs, _ = env.reset()
        total, operaciones, perdidas, terminal = 0.0, 0, [], False

        while not terminal:
            a = agente.accion(obs, eps)
            s = agente.codificar(obs)
            pos = int(obs[1])
            obs, r, terminal, _, _ = env.step(a)
            s2 = agente.codificar(obs)
            agente.buffer.push(s, a, r, s2, float(terminal))
            agente._pasos += 1
            if agente._pasos % cfg.actualiza_cada == 0:
                loss = agente.aprender()
                if loss is not None:
                    perdidas.append(loss)
            if agente._pasos % cfg.target_cada == 0:
                agente.q_fija.load_state_dict(agente.q.state_dict())
            operaciones += int(round(s2[1])) != pos
            total += r

        hist["recompensa"].append(total)
        hist["balance"].append(env.balance)
        hist["operaciones"].append(operaciones)
        hist["perdida"].append(float(np.mean(perdidas)) if perdidas else np.nan)
        hist["epsilon"].append(eps)

        if series_validacion is not None and ((ep + 1) % eval_cada == 0 or ep == 0):
            res = evaluar(agente.politica_greedy(), series_validacion)
            hist["eval_episodio"].append(ep + 1)
            hist["eval_logret"].append(float(res["logret"].mean()))
            if al_evaluar is not None:
                al_evaluar(ep + 1, hist, agente)

        if progreso is not None and time.perf_counter() - ultimo_aviso >= progreso_cada_s:
            val = ""
            if hist["eval_logret"]:
                val = f"  val {hist['eval_logret'][-1] * 100:+.1f}%"
            print(f"{progreso}  episodio {ep + 1}/{cfg.n_episodios}{val}", flush=True)
            ultimo_aviso = time.perf_counter()

    return agente, {k: np.asarray(v) for k, v in hist.items()}


def cargar_pesos(cfg, seed, pesos, escala):
    """Reconstruye el agente greedy a partir de un state_dict guardado."""
    agente = AgenteDQN(cfg, seed, escala)
    agente.q.load_state_dict(pesos)
    agente.q_fija.load_state_dict(pesos)
    agente.volcar_pesos()
    return agente


if __name__ == "__main__":
    import time

    from trader_rl import GENERADORES, generar_series

    cfg = ConfigDQN(n_episodios=2)
    assert abs(cfg.epsilon(0) - 1.0) < 1e-9
    cfg5000 = ConfigDQN()
    en_el_corte = int(cfg5000.frac_exploracion * cfg5000.n_episodios)
    assert abs(cfg5000.epsilon(en_el_corte) - cfg5000.eps_final) < 1e-6

    gen = GENERADORES["reversion"]
    val = generar_series(gen, 4, 500, seed=2024)
    t0 = time.perf_counter()
    agente, hist = entrenar(ConfigDQN(n_episodios=30), gen, seed=0, series_validacion=val, eval_cada=10)
    dt = time.perf_counter() - t0
    assert np.all(np.isfinite(hist["eval_logret"]))
    assert np.isfinite(hist["perdida"][-1])
    rng = np.random.default_rng(0)
    for eps in (0.0, 1.0):
        for pos in (0, 1):
            for _ in range(30):
                obs = np.array([rng.normal(0, 0.05), pos], np.float32)
                assert agente.accion(obs, eps) in set(ACCIONES_VALIDAS[pos])
    obs = np.array([0.02, 1.0], np.float32)
    with torch.no_grad():
        ref = agente.q(torch.tensor(agente.codificar(obs)).unsqueeze(0)).numpy()[0]
    assert np.allclose(ref, agente.valores(obs), atol=1e-5), (ref, agente.valores(obs))

    # Mismas series de test que la tabla tabular: buy & hold del cíclico es +337.5 %.
    test = generar_series(GENERADORES["ciclico"], 100, 500, seed=7777)
    bh = evaluar(politica_buy_and_hold, test)["logret"].mean() * 100
    assert abs(bh - 337.5) < 0.15, bh
    print(f"OK — 30 episodios en {dt:.1f}s ({dt / 30:.2f}s/episodio); buy & hold cíclico {bh:+.1f} %")
