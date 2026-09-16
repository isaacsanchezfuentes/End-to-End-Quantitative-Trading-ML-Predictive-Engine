# Autonomous Asynchronous Multi-Agent Trading Engine (XAUUSD) 🤖📈

## Overview
This repository showcases the core architecture of an autonomous, high-frequency quantitative trading engine optimized for the XAUUSD (Gold) market. Built in Python, the system operates on a state-of-the-art **Asynchronous Multi-Agent Framework** combined with a **Machine Learning Ensemble**, executing real-time decisions directly through the MetaTrader 5 API.

> **⚠️ Intellectual Property Notice:** To protect proprietary research, the actual Machine Learning weights (`.json` models), localized genetic agent memory states, and the specific mathematical formulas for feature engineering have been redacted from this public repository. The code provided demonstrates the system's structural engineering and execution pipeline.

---

## 🧠 Core System Architecture

The engine is built upon three foundational pillars: an evolving multi-agent ecosystem, a predictive Machine Learning pipeline, and a rigid smart-money market structure logic.

### 1. Agentic Shadow Trading & Genetic Evolution 🧬
Rather than relying on static parameters, the system deploys a continuous, decentralized ecosystem of autonomous agents (`EliteAgent`).
* **Massive Asynchronous Swarm:** A highly optimized swarm of 250 to 3,000+ localized agents is deployed across specific market regimes (Tokyo, London, NY Overlap). Thanks to its lightweight architecture, this massive parallel processing runs with near-zero computational overhead.
* **Continuous Shadow Trading:** Asynchronous agents execute and manage hypothetical trades in the background, constantly evaluating distinct environments using unique genetic parameters (WPR exhaustion, wick rejection ratios, reversal limits, and customized ATR cushions).
* **Genetic Optimization (Survival of the Fittest):** A real-time fitness function tracks the virtual PnL of every agent. The most profitable agents dynamically become the "Leaders" for their respective regimes, instantly updating the live execution thresholds of the main engine without requiring system reboots or manual backtesting.

### 2. Machine Learning Pipeline (XGBoost Ensembles) 🌳
The system does not rely on deterministic logic alone; it utilizes a probability-weighted hybrid scoring system powered by ML.
* **Asynchronous Booster Setup:** The pipeline runs decoupled evaluations for Long (Buy) and Short (Sell) conditions independently, utilizing two specialized XGBoost models per direction.
* **Feature Engineering:** Ingests live tick data and multi-timeframe arrays (M1, M5, H1) to construct synthetic variables. The models evaluate momentum vectors, acceleration curves, noise-to-strength ratios, and session-specific anomalies.
* **Probability Fusion:** The raw XGBoost predictions are mathematically fused with a hard-coded Heuristic Engine that evaluates the immediate physical properties of candlesticks (e.g., body expansion force relative to moving averages), generating a final, unified "Conviction Score."

### 3. Algorithmic Market Structure (ICT Logic) 🏛️
The ML and Agentic layers are grounded in structural market logic to prevent irrational executions:
* **Liquidity Displacement:** Algorithmic detection of Buy-Side/Sell-Side Liquidity (BSL/SSL) pools.
* **Premium/Discount Arrays:** Real-time calculation of internal structural ranges to ensure the engine only executes in statistically favorable zones.
* **Inward Hook Mechanics:** Precise detection of Williams %R (WPR) macro-exhaustion turning points.

---

## ⚡ Execution & Risk Management

Direct broker integration ensures millisecond-latency order routing with institutional-grade capital protection.
* **MT5 Python Integration:** Fully automated asynchronous order generation, modification, and telemetry logging.
* **Hard-Fuse Circuit Breakers:** Strict fiat-value emergency exits to prevent catastrophic drawdowns during black-swan events.
* **Dynamic Micro-Management:** Time-based holding buffers (90-second breathing room) and ATR-based dynamic peak-retracement trailing stops managed entirely by the active "Leader" agent.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** XGBoost (Gradient Boosting Framework)
* **Data Engineering:** Pandas, NumPy
* **Execution & Brokerage:** MetaTrader 5 API
* **Architecture:** Asynchronous Multi-Agent Design, Genetic Algorithms, Parallel Simulation
