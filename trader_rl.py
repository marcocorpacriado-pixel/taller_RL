"""
trader_rl.py — Núcleo de la práctica: agente trader con RL tabular.

Contenido (en orden):
  1. SimpleTradingEnv          copia literal del notebook del profesor.
  2. Generadores de precios    copia literal del notebook del profesor.
  3. Discretizador             tendencia continua -> bin, por cuantiles.
  4. AgenteTabular             Q-Learning y SARSA con ε-greedy.
  5. entrenar()                bucle de entrenamiento + curva de convergencia.
  6. Evaluación y baselines    agente greedy, aleatorio y Buy & Hold.

El notebook importa este módulo y se dedica sólo a experimentos y gráficas.
Las decisiones de diseño están justificadas en BITACORA.md (referencias Dx).
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# =============================================================================
# 1. Entorno (copia literal del notebook, no se modifica)
# =============================================================================
class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices):
        super(SimpleTradingEnv, self).__init__()
        self.prices = prices
        self.max_steps = len(prices) - 1

        # Acciones: 0 = Vender, 1 = Mantener, 2 = Comprar
        self.action_space = spaces.Discrete(3)

        # Estado: [tendencia_precio (-1 a 1), posicion_actual (0 o 1)]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0], dtype=np.float32),
            high=np.array([1.0, 1], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 5  # Empezar con suficiente historia para la media
        self.position = 0      # 0: sin activo, 1: con activo

        # Gestión del capital
        self.initial_balance = 1000.0
        self.balance = self.initial_balance  # Valor total del portafolio (Efectivo + Activos)
        self.cash = self.initial_balance     # Dinero en efectivo disponible
        self.shares = 0.0                    # Cantidad de acciones poseídas

        return self._get_obs(), {}

    def _get_obs(self):
        # Calcular tendencia simple basada en los últimos 5 pasos
        window = self.prices[self.current_step - 5 : self.current_step + 1]
        trend = (window[-1] - np.mean(window)) / np.mean(window)
        return np.array([np.clip(trend, -1.0, 1.0), self.position], dtype=np.float32)

    def step(self, action):
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]

        reward = 0.0
        cost_rate = 0.001  # Comisión del 0.1% por transacción

        # 1. Ejecutar acciones y actualizar efectivo/acciones
        if action == 2 and self.position == 0:  # Comprar
            # Usar todo el efectivo disponible para comprar acciones al precio anterior
            commission = self.cash * cost_rate
            effective_cash = self.cash - commission
            self.shares = effective_cash / prev_price
            self.cash = 0.0
            self.position = 1
            reward -= cost_rate

        elif action == 0 and self.position == 1:  # Vender
            # Vender todas las acciones al precio anterior
            sale_value = self.shares * prev_price
            commission = sale_value * cost_rate
            self.cash = sale_value - commission
            self.shares = 0.0
            self.position = 0
            reward -= cost_rate

        # 2. Calcular recompensa por variación del precio si mantiene la posición
        if self.position == 1:
            price_change = (curr_price - prev_price) / prev_price
            reward += price_change

        # 3. Actualizar el balance total (Net Worth) al finalizar el paso
        if self.position == 1:
            self.balance = self.cash + (self.shares * curr_price)
        else:
            self.balance = self.cash

        terminated = self.current_step >= self.max_steps
        truncated = False

        # Opcional: puedes pasar el balance actual en el diccionario de info
        info = {"balance": self.balance}

        return self._get_obs(), reward, terminated, truncated, info


# =============================================================================
# 2. Generadores de precios (copia literal del notebook)
#    Usan el RNG global de numpy: para reproducibilidad, np.random.seed() antes.
# =============================================================================
def generar_precios(n_steps=1000):
    # Combinar una tendencia suave (drift) + componente cíclico + ruido
    t = np.linspace(0, 50, n_steps)
    drift = 0.0005 * t
    ciclo = np.sin(t) * 0.05
    ruido = np.random.normal(0, 0.1, n_steps)

    # Precio base acumulado
    retornos = drift + ciclo + ruido
    precios = 100 * np.cumprod(1 + retornos)
    return precios


def generar_precios_browniano(n_steps=1000, start_price=100.0, mu=0.0001, sigma=0.02):
    dt = 1.0
    retornos = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    precios = start_price * np.cumprod(np.exp(retornos))
    return precios


def generar_precios_reversion_media(n_steps=1000, start_price=100.0, mean_price=100.0, kappa=0.05, volatility=0.02):
    precios = np.zeros(n_steps)
    precios[0] = start_price

    for t in range(1, n_steps):
        delta = kappa * (mean_price - precios[t-1]) + np.random.normal(0, volatility * mean_price)
        precios[t] = precios[t-1] + delta
        precios[t] = max(precios[t], 0.1)

    return precios


def generar_precios_regimenes(n_steps=1000, start_price=100.0):
    precios = np.zeros(n_steps)
    precios[0] = start_price
    estado = 0

    retornos = []
    for _ in range(n_steps):
        if np.random.rand() < 0.02:
            estado = 1 - estado

        if estado == 0:  # Régimen alcista
            ret = np.random.normal(0.001, 0.01)
        else:            # Régimen bajista (caídas con más ruido)
            ret = np.random.normal(-0.0015, 0.025)

        retornos.append(ret)

    retornos = np.array(retornos)
    precios = start_price * np.cumprod(1 + retornos)
    precios = np.maximum(precios, 0.1)
    return precios


GENERADORES = {
    "ciclico": generar_precios,
    "browniano": generar_precios_browniano,
    "reversion": generar_precios_reversion_media,
    "regimenes": generar_precios_regimenes,
}


def generar_series(generador, n_series, n_steps=500, seed=0):
    """Lista reproducible de series de precios (para validación y test)."""
    np.random.seed(seed)
    return [generador(n_steps) for _ in range(n_series)]


# =============================================================================
# 3. Discretización de la tendencia
# =============================================================================
def calcular_tendencias(prices):
    """Tendencia que vería el agente en cada paso (misma fórmula que _get_obs).

    tendencias[i] corresponde a current_step = i + 5 (ventana prices[i : i+6]).
    """
    ventanas = np.lib.stride_tricks.sliding_window_view(prices, 6)
    medias = ventanas.mean(axis=1)
    return np.clip((ventanas[:, -1] - medias) / medias, -1.0, 1.0)


class Discretizador:
    """Convierte la tendencia continua en un índice de bin 0..n_bins-1.

    modo="cuantiles": los cortes son los cuantiles de la tendencia medida en series
        de calibración del mismo generador, así cada bin recibe ~1/n_bins de las
        visitas (BITACORA H1, D1).
    modo="uniforme": cortes equiespaciados en [-1, 1], el rango nominal de la
        observación. Sólo se usa como comparación: casi todo cae en 2 bins.
    """

    def __init__(self, generador, n_bins, modo="cuantiles", n_series=100, n_steps=500, seed=12345):
        if modo == "uniforme":
            self.cortes = np.linspace(-1.0, 1.0, n_bins + 1)[1:-1]
        else:
            series = generar_series(generador, n_series, n_steps, seed)
            tendencias = np.concatenate([calcular_tendencias(p) for p in series])
            cuantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # n_bins - 1 cortes interiores
            self.cortes = np.quantile(tendencias, cuantiles)
        self.n_bins = n_bins

    def __call__(self, tendencia):
        return int(np.digitize(tendencia, self.cortes))


def obs_a_estado(obs, disc):
    """Observación del entorno (tendencia, posición) -> estado tabular (bin, posición)."""
    return disc(obs[0]), int(obs[1])


# =============================================================================
# 4. Agente tabular: Q-Learning / SARSA
# =============================================================================
VENDER, MANTENER, COMPRAR = 0, 1, 2

# Acciones con efecto real según la posición. Sin activo, "vender" = "mantener";
# con activo, "comprar" = "mantener". El agente sólo elige entre las que tienen
# sentido, así no gasta muestras aprendiendo Q de acciones duplicadas (D5).
ACCIONES_VALIDAS = (np.array([MANTENER, COMPRAR]),   # posición 0
                    np.array([VENDER, MANTENER]))    # posición 1


@dataclass
class Config:
    """Hiperparámetros de un experimento. Los valores por defecto son el punto de partida."""
    algoritmo: str = "qlearning"   # "qlearning" o "sarsa"
    n_bins: int = 10
    discretizacion: str = "cuantiles"  # "cuantiles" o "uniforme" (ver Discretizador)
    gamma: float = 0.95
    alpha: float = 0.05            # tasa de aprendizaje inicial
    alpha_decay: float = 1.0       # α_k = max(alpha_min, alpha · alpha_decay^k); 1.0 = constante
    alpha_min: float = 0.001
    eps_inicial: float = 1.0
    eps_final: float = 0.01        # eps_inicial == eps_final -> ε constante
    frac_exploracion: float = 0.6  # fracción de episodios en la que ε llega a eps_final
    n_episodios: int = 2000
    n_steps: int = 500             # longitud fija de las series (D2)

    def epsilon(self, episodio):
        """ε-greedy con decaimiento exponencial calibrado para llegar a eps_final
        justo en frac_exploracion · n_episodios: d = (ε_final/ε_inicial)^(1/(f·N))."""
        d = (self.eps_final / self.eps_inicial) ** (1 / (self.frac_exploracion * self.n_episodios))
        return max(self.eps_final, self.eps_inicial * d ** episodio)

    def tasa_aprendizaje(self, episodio):
        return max(self.alpha_min, self.alpha * self.alpha_decay ** episodio)


class AgenteTabular:
    """Tabla Q de forma (n_bins, 2 posiciones, 3 acciones) y su regla de actualización."""

    def __init__(self, cfg, rng):
        assert cfg.algoritmo in ("qlearning", "sarsa")
        self.cfg = cfg
        self.rng = rng
        self.Q = np.zeros((cfg.n_bins, 2, 3))

    def accion(self, estado, epsilon):
        """ε-greedy sobre las acciones válidas, con desempate aleatorio.

        Sin desempate, argmax elegiría siempre la primera acción ante empates
        (Q empieza a cero) y el agente greedy nunca compraría.
        """
        validas = ACCIONES_VALIDAS[estado[1]]
        if self.rng.random() < epsilon:
            return validas[self.rng.integers(2)]
        q = self.Q[estado][validas]
        if q[0] == q[1]:
            return validas[self.rng.integers(2)]
        return validas[q.argmax()]

    def actualizar(self, s, a, r, s_sig, a_sig, terminal, alpha):
        """Update TD: Q(s,a) <- Q(s,a) + α · [objetivo − Q(s,a)].

        Q-Learning (off-policy): objetivo = r + γ · max_a' Q(s', a')
        SARSA      (on-policy):  objetivo = r + γ · Q(s', a'), a' = acción que se tomará
        """
        if terminal:
            objetivo = r
        elif self.cfg.algoritmo == "sarsa":
            objetivo = r + self.cfg.gamma * self.Q[s_sig][a_sig]
        else:
            objetivo = r + self.cfg.gamma * self.Q[s_sig][ACCIONES_VALIDAS[s_sig[1]]].max()
        self.Q[s][a] += alpha * (objetivo - self.Q[s][a])

    def politica_greedy(self, disc):
        """Política sin exploración, como función obs -> acción (para evaluar)."""
        return lambda obs: self.accion(obs_a_estado(obs, disc), 0.0)


# =============================================================================
# 5. Entrenamiento
# =============================================================================
def entrenar(cfg, generador, disc, seed=0, series_validacion=None, eval_cada=100):
    """Entrena un agente; cada episodio usa una serie NUEVA del mismo generador.

    Devuelve (agente, historial). El historial contiene, por episodio, la
    recompensa total, el balance final y el nº de operaciones; y, cada
    `eval_cada` episodios, el log-retorno medio de la política greedy sobre las
    series de validación (curva de convergencia sin ruido de exploración, D8).
    """
    np.random.seed(seed)  # los generadores usan el RNG global
    agente = AgenteTabular(cfg, np.random.default_rng(seed))
    sarsa = cfg.algoritmo == "sarsa"
    hist = {"recompensa": [], "balance": [], "operaciones": [], "eval_episodio": [], "eval_logret": []}

    for ep in range(cfg.n_episodios):
        eps, alpha = cfg.epsilon(ep), cfg.tasa_aprendizaje(ep)
        env = SimpleTradingEnv(generador(cfg.n_steps))
        obs, _ = env.reset()
        s = obs_a_estado(obs, disc)
        a = agente.accion(s, eps)
        total, operaciones, terminal = 0.0, 0, False

        while not terminal:
            obs, r, terminal, _, _ = env.step(a)
            s_sig = obs_a_estado(obs, disc)
            # SARSA necesita a' ANTES de actualizar (es la acción que realmente tomará).
            # Q-Learning no la usa en el update; la elige después, ya con Q actualizada.
            a_sig = agente.accion(s_sig, eps) if sarsa else None
            agente.actualizar(s, a, r, s_sig, a_sig, terminal, alpha)
            if not sarsa:
                a_sig = agente.accion(s_sig, eps)
            operaciones += s_sig[1] != s[1]
            total += r
            s, a = s_sig, a_sig

        hist["recompensa"].append(total)
        hist["balance"].append(env.balance)
        hist["operaciones"].append(operaciones)

        if series_validacion is not None and ((ep + 1) % eval_cada == 0 or ep == 0):
            res = evaluar(agente.politica_greedy(disc), series_validacion)
            hist["eval_episodio"].append(ep + 1)
            hist["eval_logret"].append(res["logret"].mean())

    return agente, {k: np.array(v) for k, v in hist.items()}


# =============================================================================
# 6. Evaluación y baselines
# =============================================================================
def ejecutar_episodio(prices, politica):
    """Ejecuta un episodio sin aprender. `politica` es una función obs -> acción.

    Devuelve la trayectoria completa para poder pintar señales y balance.
    """
    env = SimpleTradingEnv(prices)
    obs, _ = env.reset()
    balances, posiciones, acciones = [env.balance], [], []
    terminal = False
    while not terminal:
        a = politica(obs)
        posiciones.append(env.position)
        acciones.append(a)
        obs, _, terminal, _, _ = env.step(a)
        balances.append(env.balance)
    return {"balances": np.array(balances), "posiciones": np.array(posiciones), "acciones": np.array(acciones)}


def evaluar(politica, series):
    """Balance final, log-retorno log(B_T/B_0) y nº de operaciones efectivas sobre varias series."""
    finales, operaciones = [], []
    for prices in series:
        ep = ejecutar_episodio(prices, politica)
        pos, acc = ep["posiciones"], ep["acciones"]
        finales.append(ep["balances"][-1])
        operaciones.append(np.sum((acc == COMPRAR) & (pos == 0)) + np.sum((acc == VENDER) & (pos == 1)))
    finales = np.array(finales)
    return {"balance_final": finales, "logret": np.log(finales / 1000.0), "operaciones": np.array(operaciones)}


def politica_aleatoria(seed=0):
    """Baseline: acción uniforme entre las 3 (equivale a env.action_space.sample())."""
    rng = np.random.default_rng(seed)
    return lambda _obs: rng.integers(3)


def politica_buy_and_hold(_obs):
    """Baseline: compra en el primer paso y después "comprar" no tiene efecto = mantener."""
    return COMPRAR


# =============================================================================
# Auto-comprobación: py trader_rl.py
# =============================================================================
if __name__ == "__main__":
    import time

    gen = GENERADORES["regimenes"]
    disc = Discretizador(gen, n_bins=10)

    # Los cuantiles reparten las visitas por igual entre bins.
    cal = np.concatenate([calcular_tendencias(p) for p in generar_series(gen, 20, seed=999)])
    frac = np.bincount([disc(x) for x in cal], minlength=10) / len(cal)
    assert np.all(np.abs(frac - 0.1) < 0.03), frac

    # calcular_tendencias reproduce exactamente la observación del entorno.
    p = generar_series(gen, 1, seed=1)[0]
    env = SimpleTradingEnv(p)
    obs, _ = env.reset()
    assert np.isclose(obs[0], calcular_tendencias(p)[0], atol=1e-6)

    # Buy & Hold a través del entorno = fórmula del notebook del profesor.
    bh = ejecutar_episodio(p, politica_buy_and_hold)["balances"][-1]
    assert np.isclose(bh, 1000 * 0.999 / p[5] * p[-1]), bh

    # La máscara de acciones: el agente greedy nunca elige una acción sin efecto.
    cfg = Config(n_episodios=300)
    val = generar_series(gen, 10, seed=2024)
    t0 = time.perf_counter()
    for algo in ("qlearning", "sarsa"):
        cfg.algoritmo = algo
        agente, hist = entrenar(cfg, gen, disc, seed=0, series_validacion=val)
        assert np.all(np.isfinite(agente.Q))
        assert np.all(agente.Q[:, 0, VENDER] == 0) and np.all(agente.Q[:, 1, COMPRAR] == 0)
        print(f"{algo:9s} val logret inicial={hist['eval_logret'][0]:+.3f}  final={hist['eval_logret'][-1]:+.3f}")
    dt = time.perf_counter() - t0
    print(f"OK — {2 * cfg.n_episodios} episodios en {dt:.1f}s ({dt / (2 * cfg.n_episodios * 495) * 1e6:.1f} µs/paso)")
