import subprocess

caminho = '/usr/bin/python3'


# Execute o arquivo script1.py
subprocess.run([caminho, "Experimentos/Experimento_citeseer.py"])
subprocess.run([caminho, 'Experimentos/Experimento_CORA.py'])
subprocess.run([caminho, 'Experimentos/Experimento_MNIST.py'])
subprocess.run([caminho, 'Experimentos/Experimento_PubMed.py'])
subprocess.run([caminho, 'Experimentos/Experimento_ionosphere.py'])