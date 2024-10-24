import pandas as pd
import matplotlib.pyplot as plt

def knn(k, df_amostra, df_entrada):
    amostra = df_amostra.iloc[:, :-1].values
    rotulos = df_amostra.iloc[:, -1].values
    entrada = df_entrada.iloc[::].values

    def distancia_euclidiana(vetor1, vetor2):
        return (sum((x - y) ** 2 for x, y in zip(vetor1, vetor2))) ** 0.5

    previsao = []

    for a in entrada:
        distancias = []

        for i in range(len(amostra)):
            dist = distancia_euclidiana(a, amostra[i])
            distancias.append((dist, rotulos[i], amostra[i][4]))  # Use o valor float corretamente aqui

        # Ordena as distâncias
        distancias.sort(key=lambda x: x[0])  # Ordena pelas distâncias

        # Pega os k vizinhos mais próximos
        vizinhos = distancias[:k]
        resultados = [v[1] for v in vizinhos]  # Rótulos dos k vizinhos
        items = [v[2] for v in vizinhos]  # Valores para desempate

        # Lógica de previsão
        if resultados.count(1) > resultados.count(0):
            previsao.append(1)
        elif resultados.count(0) > resultados.count(1):
            previsao.append(0)
        else:
            # Empate, use o valor de items para desempate
            resto = [a[4] - item for item in items]
            melhor_item = items[resto.index(min(resto))]
            previsao.append(resultados[items.index(melhor_item)])

    return previsao


def pre_processamento():
    df = pd.read_csv('Aula 18 - dataset.csv', sep=';')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    colunas_a_normalizar = df.select_dtypes(include=['float64', 'int64']).columns

    for col in colunas_a_normalizar:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)

    return df


def reamostragem(df):
    conhecimento = df.sample(frac=0.7, random_state=42)
    exemplos = df.drop(conhecimento.index)
    rotulos = df.drop(conhecimento.index)
    rotulos = rotulos.Label
    rotulos = list(rotulos)
    return conhecimento, exemplos, rotulos


def performance(knn_predictions, rotulos):
    acertos = sum(pred == rot for pred, rot in zip(knn_predictions, rotulos))
    return (acertos / len(rotulos)) * 100

def grafico(df_acuracias, k_values):
    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df_acuracias['K'], df_acuracias['Acurácia (%)'], marker='o')
    plt.title('Acurácia do KNN em Função de K')
    plt.xlabel('K (Número de Vizinhos)')
    plt.ylabel('Acurácia (%)')
    plt.xticks(k_values)
    plt.grid()
    plt.savefig('acuracia_knn.png')
    plt.show()


def main():
    acuracias = []
    k_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Valores de K desejados

    for k in k_values:
        acumulador_acuracia = 0

        for _ in range(100):  # 100 tentativas
            conhecimento, exemplos, rotulos = reamostragem(pre_processamento())
            knn_predictions = knn(k, conhecimento, exemplos)
            acumulador_acuracia += performance(knn_predictions, rotulos)

        media_acuracia = acumulador_acuracia / 100
        acuracias.append(media_acuracia)
        print(f'K={k}, Acurácia média={media_acuracia:.2f}%')

    print('======================================')

    print(f'K = {k_values[acuracias.index(max(acuracias))]} Obteve Melhor Performance')

    # Criar DataFrame para salvar as acurácias
    df_acuracias = pd.DataFrame({'K': k_values, 'Acurácia (%)': acuracias})
    df_acuracias.to_csv('acuracias_knn.csv', index=False)

    grafico(df_acuracias, k_values)

if __name__ == '__main__':
    main()