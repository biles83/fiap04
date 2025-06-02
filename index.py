import streamlit as st
from datetime import date, timedelta
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:5000/enviar-dados"

st.title("🤖 Predição de Preço de Ações")

opcoes = ["ITUB4.SA", "VALE3.SA", "PETR4.SA"]
acao = st.selectbox("Selecione a ação", opcoes)

hoje = date.today()
data_ini = st.date_input("Data Inicial")
data_fim = st.date_input("Data Final", max_value=hoje - timedelta(days=1))

if st.button("Validar e Enviar"):
    erros = []

    if data_ini > data_fim:
        erros.append("Data inicial não pode ser maior que data final.")
    if data_fim >= hoje:
        erros.append("Data final não pode ser maior ou igual à data atual.")
    if (data_fim - data_ini).days < 60:
        erros.append("O intervalo deve conter pelo menos 60 dias.")

    if erros:
        st.error("Erros encontrados:")
        for erro in erros:
            st.write(f"- {erro}")
    else:
        dados = {
            "variavel1": acao,
            "data_inicial": data_ini.isoformat(),
            "data_final": data_fim.isoformat()
        }

        try:
            r = requests.post(API_URL, json=dados)
            if r.status_code == 200:
                resultado = r.json()
                st.success(
                    f"Preço previsto para {resultado['acao']} em {resultado['data_previsao']}: R$ {resultado['preco_previsto']:.2f}")

                # Gráfico
                df_plot = pd.DataFrame(resultado['grafico'])
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_plot['data'], y=df_plot['valor_real'],
                    mode='lines+markers',
                    name='Preço Real'
                ))

                fig.add_trace(go.Scatter(
                    x=df_plot['data'], y=df_plot['valor_previsto'],
                    mode='lines+markers',
                    name='Preço Previsto',
                    line=dict(dash='dash', color='red')
                ))

                fig.update_layout(title="Gráfico: Preço Real x Previsto",
                                  xaxis_title="Data", yaxis_title="Preço (R$)")
                st.plotly_chart(fig)

            else:
                st.error("Erro da API:")
                st.text(r.text)

        except Exception as e:
            st.error("Erro ao conectar com a API.")
            st.text(str(e))
