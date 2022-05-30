# Projeto de Q-Learning e Deep Q-Learning
### Edgard Ortiz Neto
## Comparação entre Deep QLearning e Double Deep QLearning
Ambas formas de aprendizado tem em base o algoritmo do QLearning, porém, apresentam uma rede neural ao invés de uma QTable. Em relação ao Double Deep (referenciado como 'DD' nos scripts apresentados), são utilizadas duas redes neurais, uma primária e outra secundária, também conhecida como 'Target'. A finalidade de possuir duas redes neurais se deve ao fato de evitar o modelo de registrar as respectivas distorções causadas durante o aprendizado em sua rede neural, diminuindo a chance de "enganar" o modelo e que ele se torne inconsistente. Isso é feito da seguinte forma, na hora de o modelo escolher qual ação tomar, no método '*select_action()*', é preciso trocar a rede neural que servirá de base para essa decisão, no nosso caso, foi escolhido a rede secundária. Após a alteração, falta modificar o método que realiza o treinamento, para que durante o treino possa ser feita a atualização dos pesos ("weights") da rede secundária a cada X episódios (variável **refresh** utilizada no código). Depois de realizar as modificações do código do *DeepQLearning.py* é possível identificar os benefícios do Double Deep, uma vez que esse utiliza a rede primária para ir preenchendo os resuntados das ações a cada episódio e usa a rede secundária, onde será atualizado os pesos da rede com os pesos resultantes da primária a cada X episódios, para tomar a melhor decisão, aumentando a performance do modelo.
## Análise de gráficos obtidos
Foi feito 2 experimentos com o modelo "Deep QLearning", de 300 e 500 episódios, e 3 com o "Double Deep QLearning", de 300 com step de 5, 1000 com step de 15 e 2000 com step de 20 episódios. Obtiveram-se os respectivos resultados:
<br>

**Gráfico DQN 300 episódios**
![alt text](https://github.com/Edortizneto/DDQN/blob/main/imgs/DQN300.jpg?raw=true)]
Nessa versão do Deep QLearning treinada com 300 episódios, é possível observar que o modelo oscila até atingir a marca superior a 200 de *reward* e continua oscilando, mesmo que mantendo como base a marca superior a 200, até acabar os 300 episódios.
<br>

**Gráfico DQN 500 episódios**
![alt text](https://github.com/Edortizneto/DDQN/blob/main/imgs/DQN500.jpg?raw=true)
Nessa versão do Deep QLearning treinada com 500 episódios, é possível observar que o modelo oscila até atingir a marca superior a 250 de *reward* e continua oscilando, mesmo que mantendo como base a marca superior a 200, até chegar aos 300 episódios. Porém, após os 300 o modelo parece que perde tudo que aprendeu e volta para uma média de *rewards* bastante negativa até oscilar de volta à uma marca mais positiva próxima ao episódio de número 500. 
<br>

**Gráfico DDQN 300 episódios**
![alt text](https://github.com/Edortizneto/DDQN/blob/main/imgs/DDQN300.jpg?raw=true)
Nesta versão do Double Deep QLearning treinada com 300 episódios, é possível observar que o modelo oscila bastante até uma marca de *reward* desejada, porém, sua amplitude de oscilação é menor, devido à sua rede mais "conservadora" de tomada de decisões.
<br>

**Gráfico DDQN 1000 episódios**
![alt text](https://github.com/Edortizneto/DDQN/blob/main/imgs/DDQN1000.jpg?raw=true)
Nesta versão do Double Deep QLearning treinada com 1000 episódios, é possível observar que o modelo oscila bastante até uma marca de *reward* desejada, acabando por não conseguir manter uma média de bons resultados.
<br>

**Gráfico DDQN 2000 episódios**
![alt text](https://github.com/Edortizneto/DDQN/blob/main/imgs/DDQN2000.jpg?raw=true)
Nesta versão do Double Deep QLearning treinada com 2000 episódios, é possível observar que o modelo oscila bastante até uma marca de *reward* desejada, acabando por convergir apenas no final dos 2000 episódios para um *reward* mais próximo do desejado.

## Análise dos resultados encontrados e Conclusão
Foi desenvolvido um código *Tournament.py* a fim de comparar a média com desvio padrão de 10 amostras entre dois modelos. Dado que os melhores modelos, entre DQN e DDQN, foram de 300 e 300 episódios respectivamente, obteve-se esse resultado:
```
DQN
Score = 281.07014107761137
Score = 270.3249082504195
Score = 266.80065319570497
Score = 303.0350532454352
Score = 240.14676334512183
Score = 278.85535593444894
Score = 244.31268226197597
Score = 281.78173713371086
Score = 255.0531544153119
Score = 272.6353329753034
DDQN
Score = 171.9081229705464
Score = 238.2888335922226
Score = 176.5780558254133
Score = -252.19739812910777
Score = 214.64482495196143
Score = 288.55813556640146
Score = 247.88006099744334
Score = -85.78731016130716
Score = 235.71681952677955
Score = 245.26200927776176
Deep QLearning 
 mean = 269.4015781835044        std = 17.942287835811136
Double Deep QLearing
 mean = 148.0852154418115        std = 166.03706091216665
```
Portanto, conclui-se que o Deep QLearning, ainda foi mais regular e melhor do que o Double Deep.
Porém, foi visto que no modelo de 1000 episódios com atualização a cada 15, a nave sempre pousa de maneira correta, obtendo-se esses resultados:
```
DDQN
Score = 48.30065994858457
Score = 46.92741319741548
Score = 38.553808509816996
Score = 83.21306698778672
Score = 190.6932457139319
Score = 71.89620506986567
Score = 63.95777835252631
Score = 60.25197735593924
Score = 205.95809962912995
Score = 245.88441295917792        
Double Deep QLearning 
 mean = 105.56366677241746       std = 73.23989957338223
```
Os números parecem ruins, mas é necessário pontuar um problema encontrado no ambiente: Algumas vezes, a nave consegue pousar todavia não é registrado no ambiente como estado terminal, logo, o agente acha que ainda é necessário realizar ajustes e acaba consumindo todos os stes (máximo de 3000), tomando uma pena no score e diminuindo para números próximos de 50.0.


No final das contas, é preciso ponderar que apesar dos resultados do Deep QLearning serem melhores, o modelo de Double Deep QLearning também pode ser competente.