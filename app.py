import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- 1. CONFIGURAÇÃO E AUTENTICAÇÃO DO GEMINI ---

def configurar_gemini():
    """Tenta configurar o cliente Gemini lendo a chave da variável de ambiente."""
    # O Streamlit Cloud injeta a chave configurada nos Secrets como variável de ambiente
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

    if not GEMINI_API_KEY:
        # Mensagem de erro que será exibida no Streamlit se a chave falhar
        st.error("ERRO: A variável 'GEMINI_API_KEY' não está configurada. Configure o Secret no Streamlit Cloud.")
        return None

    try:
        # Inicialização do cliente
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
    baseado em valores ilustrativos da Tabela de Honorários Mínimos da OAB/RJ.
    
    ATENÇÃO: Estes valores devem ser substituídos pelos valores ATUAIS da tabela da OAB/RJ.
    """
    
    Tabela_Pisos_OABRJ = {
        "Cível Comum (Conhecimento)": 6500.00,  # Exemplo: Ações Ordinárias (OAB/RJ item 1.2.1/1.2.2)
        "Família (Divórcio Consensual)": 4000.00, # Exemplo: Divórcio Consensual
        "Trabalhista (Reclamante)": 3000.00,     # Exemplo: Reclamação Trabalhista sem vínculo/baixo valor
        "Previdenciário (Administrativo)": 2500.00,
        "Outro": 3000.00
    }
    
    piso_fixo = Tabela_Pisos_OABRJ.get(tipo_acao, 3000.00)
    percentual_oab = 0.20 * valor_causa 
    
    # Adota o maior entre o piso fixo e 10% do valor da causa (como regra simples de contingência)
    return max(piso_fixo, percentual_oab * 0.5)


# --- 3. FUNÇÃO PRINCIPAL DE CÁLCULO COM GEMINI ---

def calcular_honorarios_com_gemini(cliente, valor_causa, tipo_acao, complexidade, fase_processual):
    if not cliente:
        return None

    percentual_base = 0.20
    calculo_base = valor_causa * percentual_base

    valor_minimo_oabrj = obter_valor_minimo_oabrj(tipo_acao, valor_causa)
    calculo_base_ajustado = max(calculo_base, valor_minimo_oabrj)

    # O prompt deve ser muito claro e exigir um formato de saída exato (JSON ou delimitador)
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
    
    try:
        response = cliente.models.generate_content(
            model=MODELO_GEMINI,
            contents=contexto,
            temperature=0.3
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
                        # Log de erro (útil para debug no Streamlit Cloud logs)
                        print(f"Erro de conversão de valor na linha: {linha}")
                        continue
        
        justificativa = partes[1].replace("JUSTIFICATIVA:", "").strip() if len(partes) > 1 else "Não foi possível gerar a justificativa da IA."
        
        return {
            "piso_oabrj": valor_minimo_oabrj,
            "base": calculo_base_ajustado,
            "minimo": valores.get('MINIMO'),
            "medio": valores.get('MEDIO'),
            "maximo": valores.get('MAXIMO'),
            "justificativa": justificativa
        }

    except APIError as e:
        st.error(f"Erro na API do Gemini. Tente novamente: {e}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro no processamento da resposta da IA: {e}")
        return {"piso_oabrj": valor_minimo_oabrj, "base": calculo_base_ajustado, 
                "minimo": None, "medio": None, "maximo": None, 
                "justificativa": "Falha na comunicação com a IA. Cálculo baseado apenas em regras simples internas."}

# --- 4. FUNÇÃO DE GERAÇÃO DE GRÁFICO (RELATÓRIO VISUAL) ---

def gerar_grafico(resultados):
    """Gera um gráfico de barras comparando os valores."""
    
    valores = [
        resultados.get('piso_oabrj'), 
        resultados.get('minimo'), 
        resultados.get('medio'), 
        resultados.get('maximo')
    ]
    
    titulos = ['Piso OAB/RJ', 'Sugestão Mínima', 'Sugestão Média', 'Sugestão Máxima']
    cores = ['#D9534F', '#F0AD4E', '#5CB85C', '#5BC0DE']
    
    dados_grafico = [(t, v, c) for t, v, c in zip(titulos, valores, cores) if v is not None]
    
    if not dados_grafico:
        return "Não há dados suficientes para gerar o gráfico."

    titulos_f, valores_f, cores_f = zip(*dados_grafico)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(titulos_f, valores_f, color=cores_f)
    
    ax.set_title('Faixa de Honorários Sugerida vs. Piso OAB/RJ', fontsize=16)
    ax.set_ylabel('Valor (R$)', fontsize=14)
    ax.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=15)
    
    max_val = max(valores_f) if valores_f else 0
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max_val*0.01), 
                f'R$ {yval:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.'), 
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


# --- 5. INTERFACE STREAMLIT (Lógica Principal de Exibição) ---

# Configuração da página deve ser a primeira chamada ao Streamlit
st.set_page_config(page_title="Calculadora de Honorários Advocatícios (OAB/RJ + Gemini)", layout="wide")

st.title("⚖️ Calculadora de Honorários OAB/RJ + Gemini")
st.markdown("Uma ferramenta para advogados, baseada em regras de mercado e na Tabela Mínima da OAB/RJ, com apoio da IA para contextualização.")
st.markdown("---")

# Inicializa o cliente Gemini apenas uma vez
CLIENTE_GEMINI = configurar_gemini()

# Seção de Entrada de Dados
st.header("1. Informações do Caso")

col1, col2 = st.columns(2)

with col1:
    valor_causa_input = st.number_input("Valor da Causa (R$):", min_value=100.00, value=10000.00, step=1000.00)
    
with col2:
    tipo_acao = st.selectbox(
        "Tipo de Ação (Ref. OAB/RJ):",
        ["Cível Comum (Conhecimento)", "Família (Divórcio Consensual)", 
         "Trabalhista (Reclamante)", "Previdenciário (Administrativo)", "Outro"]
    )

complexidade = st.select_slider(
    "Complexidade do Caso (Avaliação do Advogado):",
    options=['Baixa', 'Média', 'Alta'],
    value='Média'
)

fase_processual = st.selectbox(
    "Fase Processual Atual:",
    ["Fase de Conhecimento (Inicial)", "Fase de Instrução", "Fase Recursal (Tribunal Local)", "Fase de Execução"]
)

st.markdown("---")

if st.button("Calcular Honorários com IA", type="primary"):
    if valor_causa_input < 100:
        st.error("Por favor, insira um Valor da Causa válido.")
    elif not CLIENTE_GEMINI:
        # Se a chave falhou, a mensagem de erro já está sendo exibida no topo
        st.warning("Falha ao inicializar o cliente Gemini. Verifique a chave de API nos Secrets do Streamlit Cloud.")
    else:
        with st.spinner("Processando informações e consultando a inteligência artificial..."):
            
            # Chama a função de cálculo
            resultados = calcular_honorarios_com_gemini(
                CLIENTE_GEMINI,
                valor_causa_input, 
                tipo_acao, 
                complexidade, 
                fase_processual
            )

        if resultados:
            st.success("✅ Cálculo Concluído!")
            st.header("2. Resultados e Análise")
            
            # Exibição dos Resultados em Cards
            col_min, col_medio, col_max = st.columns(3)

            def formatar_valor(valor):
                # Formatação BR (Ponto para milhar, vírgula para decimal)
                return f"R$ {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') if valor is not None else "N/A"

            col_min.metric("Mínimo Sugerido", formatar_valor(resultados.get('minimo')), 
                           f"Piso OAB/RJ: {formatar_valor(resultados.get('piso_oabrj'))}")
            col_medio.metric("Médio Sugerido", formatar_valor(resultados.get('medio')), 
                             f"Base 20%: {formatar_valor(resultados.get('base'))}")
            col_max.metric("Máximo Sugerido", formatar_valor(resultados.get('maximo')))

            st.markdown("### 📊 Relatório Visual")
            
            # Geração do Gráfico
            fig = gerar_grafico(resultados)
            if isinstance(fig, str):
                st.warning(fig)
            else:
                st.pyplot(fig)
            
            st.markdown("### 🤖 Justificativa da Inteligência Artificial (Gemini)")
            st.info(resultados['justificativa'])

st.markdown("---")
st.caption("Projeto desenvolvido para fins educacionais. Os valores da Tabela OAB/RJ são ilustrativos e devem ser confirmados com a versão atualizada da seccional.")
