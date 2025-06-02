from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from lstm_model import PredictionModel
from sklearn.preprocessing import StandardScaler
from prometheus_flask_exporter import PrometheusMetrics
import psutil
from prometheus_client import start_http_server, Summary, Gauge
from loguru import logger
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)

# === Métricas Prometheus ===
REQUEST_TIME = Summary('response_time_seconds', 'Tempo de resposta da API')
CPU_USAGE = Gauge('cpu_usage_percent', 'Uso de CPU')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Uso de memória RAM')

# Carregar o Modelo
model = PredictionModel(input_dim=1, hidden_dim=32,
                        num_layers=2, output_dim=1).to(device)
model.load_state_dict(torch.load('modelo.pth'))
model.eval()

scaler = StandardScaler()

# === Inicializa servidor de métricas Prometheus ===
start_http_server(8001)  # Prometheus coleta em :8001/metrics


@REQUEST_TIME.time()
@app.route('/enviar-dados', methods=['POST'])
def prever():
    inicio = time.time()
    dados = request.get_json()
    ticker = dados["variavel1"]
    data_ini = dados["data_inicial"]
    data_fim = dados["data_final"]

    try:
        logger.info(
            f"Recebendo predição para {ticker} ({data_ini} → {data_fim})")

        # Coleta preços
        df = yf.download(ticker, start=data_ini, end=data_fim)
        if df.empty:
            return jsonify({"erro": "Sem dados para o período."}), 400

        serie_original = df[['Close']].copy()
        dados_norm = scaler.fit_transform(serie_original)

        if len(dados_norm) < 60:
            return jsonify({"erro": "É necessário ao menos 60 dias de dados para previsão."}), 400

        # Últimos 60 dias
        entrada = dados_norm[-60:]
        entrada_tensor = torch.tensor(
            entrada).float().unsqueeze(0)

        with torch.no_grad():
            saida_norm = model(entrada_tensor)
            saida = scaler.inverse_transform(saida_norm.numpy())

        # Gráfico: últimos 60 reais + 1 predição
        historico = serie_original[-60:].reset_index()
        historico.columns = ['data', 'valor_real']
        historico['valor_previsto'] = np.nan

        data_pred = pd.to_datetime(data_fim)
        nova_linha = pd.DataFrame({'data': [data_pred], 'valor_real': [
                                  np.nan], 'valor_previsto': [saida[0][0]]})

        grafico_df = pd.concat([historico, nova_linha], ignore_index=True)

        tempo_total = time.time() - inicio
        logger.info(
            f"Predição concluída em {tempo_total:.3f}s | Preço previsto: {saida[0][0]:.2f}")

        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

        return jsonify({
            "status": "sucesso",
            "preco_previsto": float(saida[0][0]),
            "grafico": grafico_df.to_dict(orient="records"),
            "acao": ticker,
            "data_previsao": data_fim
        })

    except Exception as e:
        logger.exception("Erro ao processar a predição:")
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500


if __name__ == '__main__':
    logger.info("API iniciada com monitoramento Prometheus")
    app.run(debug=True)
