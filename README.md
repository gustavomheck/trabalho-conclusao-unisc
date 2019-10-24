# trabalho-conclusao-unisc

Código fonte do Trabalho de Conclusão "Cálculo da diferença de sobreposição de quadros em streaming de vídeo para medição de velocidade em tempo real".

Dependências
-----

Os seguintes pacotes são necessários para executar o programa:

* tensorflow-gpu==1.14.0
* tensornets==0.4.1
* opencv-contrib-python
* numpy
* imutils

Uso
-----

Há dois arquivos que podem ser executados:

```
speed_measurement.py # arquivo principal
speed_measurement_tests.py # testes unitários
```

Argumentos:

```
-i
--input
O caminho do arquivo de entrada

-o
--output 
Salvar cada quadro e as predições na pasta /output (deve ser criada manualmente)

-d
--distance
A distância máxima para associar dois objetos, o padrão é 30

-c
--confidence 
A confiança mínima da predição, o padrão é 0.75
```