import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from google import genai
from google.genai.errors import APIError
from google.genai import types # Importação necessária para GenerationConfig


# --- 1. CONFIGURAÇÃO E AUTENTICAÇÃO DO GEMINI ---

def configurar_gemini():
    """Tenta configurar o cliente Gemini lendo a chave da variável de ambiente (Secrets do Streamlit Cloud)."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

    if not GEMINI_API_KEY:
        st.error("ERRO: A variável 'GEMINI_API_KEY' não está configurada. Configure o Secret no Streamlit Cloud.")
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client
    except Exception as e:
        st.error(f"Erro ao inicializar o cliente Gemini: {e}")
        return None

MODELO_GEMINI = 'gemini-2.5-flash'


# --- 2. LÓGICA DE NEGÓCIO (Tabela OAB/RJ - Mantenha Atualizada!) ---

def obter_valor_minimo_oabrj(tipo_acao, valor_causa):
    """
    Retorna o valor MÍNIMO de honorários para o tipo de ação, 
    baseado em valores ILUSTRATIVOS da Tabela de Honorários Mínimos da OAB/RJ.
    """
    
    # ATENÇÃO: SUBSTITUA ESTES VALORES ILUSTRATIVOS PELOS VALORES ATUAIS DA TABELA OAB/RJ.
    Tabela_Pisos_OABRJ = {
        "Cível Comum (Conhecimento)": 6500.00,
        "Família (Divórcio Consensual)": 4000.00,
        "Trabalhista (Reclamante)": 3000.00,
        "Previdenciário (Administrativo)": 2500.00,
        "Outro": 3000.00
    }
    
    piso_fixo = Tabela_Pisos_OABRJ.get(tipo_acao, 3000.00)
    percentual_oab = 0.20 * valor_causa 
    
    # Regra base: Mínimo entre o Piso Fixo e 10% do Valor da Causa
    return max(piso_fixo, percentual_oab * 0.5)


# --- 3. FUNÇÃO PRINCIPAL DE CÁLCULO COM GEMINI ---

def calcular_honorarios_com_gemini(cliente, valor_causa, tipo_acao, complexidade, fase_processual):
    if not cliente:
        return None

    percentual_base = 0.20
    calculo_base = valor_causa * percentual_base

    valor_minimo_oabrj = obter_valor_minimo_oabrj(tipo_acao, valor_causa)
    calculo_base_ajustado = max(calculo_base, valor_minimo_oabrj)

    # Criação do Prompt Contextualizado
    contexto = f"""
Você é um consultor de honorários advocatícios no Rio de Janeiro (OAB/RJ) e deve sugerir uma faixa de valores justa e razoável.
Seu objetivo é gerar a sugestão mais ética e justa.

CONTEXTO DO CASO:
- Tipo de Ação: {tipo_acao}
- Valor da Causa: R$ {valor_causa:,.2f}
- Complexidade (auto-avaliada pelo usuário): {complexidade}
- Fase Processual: {fase_processual}

DIRETRIZES TÉCNICAS:
1. O valor MÍNIMO legal/ético para esta ação é de **R$ {valor_minimo_oabrj:,.2f}**, conforme o piso ilustrativo da OAB/RJ.
2. O valor base calculado por uma regra simples de 20% é de **R$ {calculo_base_ajustado:,.2f}**.

SUGESTÃO DA IA:
- Sugira uma FAixa de Honorários Contratuais (Mínimo, Médio e Máximo).
- O valor Mínimo sugerido deve ser SEMPRE igual ou superior ao piso da OAB/RJ (R$ {valor_minimo_oabrj:,.2f}).
- O valor Médio deve ser um ajuste razoável do valor base (R$ {calculo_base_ajustado:,.2f}), considerando a Complexidade e a Fase.
- O valor Máximo deve representar o limite superior para um caso de sucesso e alta demanda.

A resposta deve ser formatada **EXATAMENTE** desta forma, usando o delimitador '---' para separar valores e justificativa:
MINIMO: [Valor em reais, sem R$ e com ponto como separador decimal. Ex: 6500.00]
MEDIO: [Valor em reais, sem R$ e com ponto como separador decimal]
MAXIMO: [Valor em reais, sem R$ e com ponto como separador decimal]
---
JUSTIFICATIVA: [Texto conciso e profissional em português, explicando a sugestão com base na complexidade e nas referências da OAB/RJ.]
"""
    
    # CORREÇÃO: Usar GenerationConfig para passar temperature
    config = types.GenerateContentConfig(
        temperature=0.3
    )

    try:
        response = cliente.models.generate_content(
            model=MODELO_GEMINI,
            contents=contexto,
            config=config # Uso do parâmetro corrigido
        )
        
        # Processamento da Resposta
        resultado_bruto = response.text.strip()
        partes = resultado_bruto.split("---")
        
        valores = {}
        if len(partes) > 0:
            for linha in partes[0].strip().split('\n'):
                if ":" in linha:
                    chave, valor_str = linha.split(":")
                    try:
                        valor_str_limpo = valor_str.strip().replace("R$", "").replace(",", "").replace(" ", "")
                        if valor_str_limpo.endswith('.'):
                            valor_str_limpo = valor_str_limpo[:-1]
                            
                        if valor_str_limpo:
                            valor_numerico = float(valor_str_limpo) 
                            valores[chave.strip()] = valor_numerico
                        
                    except ValueError:
                        print(f"Erro de
