import os

pastas = [
'citeseer',
'cora',
'Ionosphere', 
'MNIST', 
'twitch'
]

# Especifique o caminho da pasta onde estão os arquivos .py que você deseja executar
pasta = '/home/ledzeppelin/Desktop/experimentos/experimentos_pesquisa/Experimentos'

for dataset in pastas:
    caminho = pasta + '/' + dataset
    arquivos = os.listdir(caminho)
    # print(arquivos)
    # print(caminho)
    for arquivo in arquivos:
        executavel = caminho + '/' + arquivo
        print(executavel)
        exec(open(executavel).read())
        
    