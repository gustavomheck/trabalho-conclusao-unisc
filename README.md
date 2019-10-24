Dependências
-----

Os seguintes pacotes são necessários para executar o programa:

* tensorflow==1.14.0
* tensornets==0.4.1
* opencv-contrib-python
* numpy
* imutils

Uso
-----

Há dois arquivos que podem ser executados. O primeiro recebe um vídeo como entrada e realiza a medição de velocidade dos veículos e o segundo contém testes unitários.

python speed_measurement.py
python speed_measurement_tests.py

Argumentos:

```
--input # caminho do arquivo de entrada
--output # salvar cada quadro e as predições na pasta /output (deve ser criada manualmente)
--distance # distância máxima para associar dois objetos, o padrão é 30
--confidence # confiança mínima da predição, o padrão é 0.75
```