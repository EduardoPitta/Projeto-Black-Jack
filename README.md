---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3.9.0 64-bit
    name: python3
---

# Projeto: Será que Machine Learning pode quebrar a banca? #

<p align="center">
  <img src="pexels-photo-3279691.jpeg" >
</p>

Blackjack é um dos jogos mais simples que você pode jogar em um Casino.
As regras são simples, você joga contra a banca e ambos começam com duas cartas, a soma destas cartas precisa ser abaixo ou igual a 21, sendo que o jogador mais perto de 21 ganha e quem tiver mais que 21 "estoura" e automaticamente perde.
Para começar um projeto de Machine Learning preciso criar um jogo de Blackjack em python e simular um jogador usando diferentes estratégias de jogo.
Este projeto vai ficar mais complexo com o tempo, adicionando mais de um jogador, mistura de estratégias, apostas e finalmente a construção de um modelo de ML que vai jogar do melhor jeito possível de acordo com as cartas na mesa e a estratégia adotada.
Também vou analisar os dados explorando as variáveis relacioandas ao jogo, como quantidade de baralhos em nosso deck, quantidade de jogadores, etc...

Me sinto na obrigação de dizer que **esse projeto visa somente avaliar as capacidades da Inteligência Artificial em um jogo de azar. Jogos de azar não são investimento e cassino não é caridade. A banca SEMPRE tem a vantagem mesmo usando várias 'técnicas' e macetes. Esse código NÃO deve ser usado em cassinos e só tem fins acadêmicos.**

Primeiro, precisamos programar o jogo:


Fazendo as funções do jogo:

```python
import pandas as pd
import numpy as np
import random

#Checa a mão somando os valores e dá como output o valor da soma
def sum_hand(hand):
    d_val = {'2': 2, '3': 3, '4': 4, '5': 5,
     '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
     'J': 10, 'Q': 10, 'K': 10, 'A': 11, 'A.': 1}

    soma = sum(d_val[str(k)] for k in hand)
    return soma
#Cria o nosso "shoe", que é o baralho que vai ser utilizado
def create_shoe(n_decks=6):
    deck = [2,3,4,5,6,7,8,9,10,'J','Q','K','A']
    shoe = deck*4*n_decks
    random.shuffle(shoe)
    return shoe

#Função de comprar cartas
def deal_cards(hand,shoe,n_cards=1):
    for i in range(n_cards):
        hand.append(shoe.pop())
    return hand

#Checa se você estourou ou automaticamente ganhou
def check_hand(hand):
    if sum_hand(hand) > 21:
        hand = ace_of_spades(hand)
        if sum_hand(hand) > 21:
            return 'bust'
        else:
            return 'keep'
    elif sum_hand(hand) == 21:
        return 'win'
    else:
        return 'keep'

#Imprime os resultados, somente para conferencia
def print_results(d_hand,p_hand):
    return
    #print('\n Mão da banca: ', d_hand,' valendo',sum_hand(d_hand), ' pontos.' '\n Sua mão:', p_hand , ' valendo', sum_hand(p_hand) , ' pontos')

#Se você estoura e tiver um Ás na mão este Ás vale 1 ao invés de 11

def ace_of_spades(hand):
    if sum_hand(hand) > 21 and 'A' in hand:
        hand[hand.index('A')] = 'A.'
    return hand

def count_cards(p_hand,d_hand,strategy,r_count):
    r_count += card_counter(p_hand,strategy)
    r_count += card_counter(d_hand,strategy)
    return r_count



```

Código para contagem de cartas com oito estratégias de contagem de cartas:
Hi-Lo
Hi-opt I
Hi-Opt II
KO
Omega II
Red 7
Halves
Zen Count
Sem estratégia

```python
import pandas as pd
#Fazer um dataframe com a contagem de cada carta de acordo com a estratégia
vals = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, 
            '7': 0, '8': 0, '9': 0, '10': -1, 'J': -1, 
            'Q': -1, 'K': -1, 'A': -1, 'A.':-1}

df = pd.DataFrame(vals, index=[0])

df.loc[len(df), :] = [0,1,1,1,1,0,0,0,-1,-1,-1,-1,0,0]
df.loc[len(df), :] = [1,1,2,2,1,1,0,0,-2,-2,-2,-2,0,0]
df.loc[len(df), :] = [1,1,1,1,1,1,0,0,-1,-1,-1,-1,-1,-1]
df.loc[len(df), :] = [1,1,2,2,2,1,0,-1,-2,-2,-2,-2,0,0]
df.loc[len(df), :] = [1,1,1,1,1,0,0,0,-1,-1,-1,-1,-1,-1]
df.loc[len(df), :] = [.5,1,1,1.5,1,.5,0,-.5,-1,-1,-1,-1,-1,-1]
df.loc[len(df), :] = [1,1,2,2,2,1,0,0,-2,-2,-2,-2,-1,-1]
df.loc[len(df), :] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

df.rename({0: 'Hi-Lo',
           1: 'Hi-Opt I',
           2: 'Hi-Opt II',
           3: 'KO',
           4: 'Omega II',
           5: 'Red 7',
           6: 'Halves',
           7: 'Zen Count',
           8: 'No Strategy'}, inplace=True)

#Para uso posterior vamos usar pickle
df.to_pickle('Card_Counting_Values')

# Conta as cartas de uma mão
def card_counter(hand, strategy='Hi-Lo'):
    
    df = pd.read_pickle('Card_Counting_Values')
    card_sum = sum([df.loc[strategy][str(i)].item() for i in hand])
    return card_sum
    
# Função recursiva que pergunta ao jogador se deseja comprar uma carta ou continuar com a mão
def ask_input(d_hand,p_hand,shoe,r_count,strategy):
    #print("\nQual será sua jogada? (Hit ou Stay)")

    jogada = player_move(p_hand,15,r_count,d_hand)

    if jogada == 'hit':
        p_hand = deal_cards(p_hand,shoe,1)

        if check_hand(p_hand) == 'bust':
            #print('Você estourou a mão...')
            print_results(d_hand,p_hand)
            r_count = count_cards(p_hand,d_hand,strategy,r_count)
            return ['loss',p_hand,d_hand,r_count,strategy]

        elif check_hand(p_hand) == 'win':
    
            #print('21! você ganhou!')
            print_results(d_hand,p_hand)
            r_count = count_cards(p_hand,d_hand,strategy,r_count)
            return ['win', p_hand, d_hand, r_count,strategy]

        elif check_hand(p_hand) == 'keep':
            
            return ask_input(d_hand,p_hand,shoe,r_count,strategy)
            
    if jogada == 'stay':
       
        return dealer_turn(p_hand, d_hand, r_count, shoe, strategy)
    
```

Algoritmo de jogador, define como o jogador jogará.

```python
def player_move(your_hand, limit, true_cnt, dealer_hand):
    """
    Chooses 'hit' or 'stay' depending on the limit set and count
    """
    
    dtotal = sum_hand(dealer_hand[:1])

    # Se tiver bastante cartas boas, se arrisca mais
    if true_cnt > 0:
        if sum_hand(your_hand) >= limit:
            return 'stay'
        elif sum_hand(your_hand) < limit:
            return 'hit'
        elif dtotal >= 10:
            return 'stay'
        
        
    # Cartas ruins, vai ser mais convervador
    elif true_cnt < 0:
        if sum_hand(your_hand) <= limit:
            return 'hit'
        elif sum_hand(your_hand) > limit:
            return 'stay'
        elif dtotal < 10:
            return 'hit'
        
        
    # Neutro, jogar uma estratégia genérica    
    else:
        if sum_hand(your_hand) >= 17:
            return 'stay'
        elif sum_hand(your_hand) < 17:
            return 'hit'
```

Algoritmo da banca

```python
def dealer_turn(p_hand, d_hand, r_count, shoe, strategy): 
   
   #Ativado no turno da banca, a banca joga e checa o resultado do jogo
    if sum_hand(d_hand) > sum_hand(p_hand):
        #print('A banca ganhou...')
        r_count = count_cards(p_hand,d_hand,strategy,r_count)
        print_results(d_hand,p_hand)
        return ['loss', p_hand, d_hand, r_count, strategy]
        
    elif sum_hand(d_hand) < sum_hand(p_hand) and sum_hand(d_hand) >=17 :
        #print('Você ganhou!')
        r_count = count_cards(p_hand,d_hand,strategy,r_count)
        print_results(d_hand,p_hand)
        return ['win', p_hand, d_hand, r_count, strategy]

    elif sum_hand(d_hand) == sum_hand(p_hand) and sum_hand(d_hand) >= 17 :
        #print('Empate! Mesma pontuação.')
        r_count = count_cards(p_hand,d_hand,strategy,r_count)
        print_results(d_hand,p_hand)
        return ['draw', p_hand, d_hand, r_count,strategy]
    
    elif sum_hand(d_hand) <= sum_hand(p_hand) and sum_hand(d_hand) < 17 :
        while sum_hand(d_hand) < 17:
            d_hand = deal_cards(d_hand,shoe,1)
            if check_hand(d_hand) == 'bust':
                #print('Você venceu! A banca estourou.')
                r_count = count_cards(p_hand,d_hand,strategy,r_count)
                print_results(d_hand,p_hand)
                return ['win', p_hand, d_hand, r_count,strategy]

            elif sum_hand(d_hand) > sum_hand(p_hand):
                #print('A banca ganhou...')
                r_count = count_cards(p_hand,d_hand,strategy,r_count)
                print_results(d_hand,p_hand)
                return ['loss', p_hand, d_hand, r_count,strategy]

            elif sum_hand(d_hand) == sum_hand(p_hand):
                #print('Empate! Ambos tem a mesma pontuação.')
                r_count = count_cards(p_hand,d_hand,strategy,r_count)
                print_results(d_hand,p_hand)
                return ['draw', p_hand, d_hand, r_count,strategy]
            
            elif sum_hand(d_hand) < sum_hand(p_hand):
                #print('O jogador ganhou!')
                r_count = count_cards(p_hand,d_hand,strategy,r_count)
                print_results(d_hand,p_hand)
                return ['win', p_hand, d_hand, r_count,strategy]

    
```

Algoritmo do jogo

```python
def blackjack(shoe,r_count,strategy):
    
    p_hand = []
    d_hand = []
    p_hand = deal_cards(p_hand,shoe,2)
    d_hand = deal_cards(d_hand,shoe,2)
    
    if check_hand(d_hand) == 'win':
        #print("\n A banca ganha com um Blackjack na primeira!")
        print_results(d_hand,p_hand)
        r_count = count_cards(p_hand,d_hand,strategy,r_count)
        return ['loss',p_hand,d_hand,r_count,strategy]

    elif check_hand(p_hand) == 'win': 
        #print("\n O jogador ganha com um Blackjack na primeira!")
        print_results(d_hand,p_hand)
        r_count = count_cards(p_hand,d_hand,strategy,r_count)
        return ['win', p_hand, d_hand, r_count,strategy]

    elif check_hand(p_hand) == 'win' and check_hand(d_hand) == 'win':
        #print('Um empate! Dois Blackjack de primeira!')
        print_results(d_hand,p_hand)
        r_count = count_cards(p_hand,d_hand,strategy,r_count)
        return ['draw',p_hand,d_hand,r_count,strategy]

    return ask_input(d_hand, p_hand, shoe, r_count, strategy)
```

Executando diferentes estratégias no jogo.
10000 turnos para cada estratégia

```python
f = open('data.csv','w')
f.write('w or l,p_hand,d_hand,r_count,strategy\n')

strats = list(pd.read_pickle('Card_Counting_Values').index)
results = []
for strategy in strats:
    i = 0
    r_count = 0
    shoe = create_shoe()
    while i <10000:
        if len(shoe) <= 52:
            shoe=create_shoe()
            r_count = 0
        out = blackjack(shoe,r_count, strategy)
        r_count = out[3]
        out = ', '.join(map(str,out))
        results.append(out)
        f.write(str(out))
        f.write('\n')
        i += 1
f.close()

exit()

```

Agora vamos dar uma olhada na razão win/lose das estratégias.
Vou considerar draw (empate) como uma vitoria pois não perdemos dinheiro quando empatamos.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv
import collections, numpy

df = pd.read_csv('data.csv')
df_list = [x[1] for x in df.groupby('strategy', sort=False)]

fig , ax = plt.subplots(nrows = 3, ncols = 3)

sns.countplot(data = data, x = 'Age', ax=ax[0][0])
sns.countplot(x='Employment Type', data=data, ax=ax[0][1])
sns.countplot(data = data, x = 'GraduateOrNot', ax=ax[0][2])
sns.histplot(data = data, x = 'AnnualIncome', ax=ax[1][0])
sns.histplot(data = data, x = 'FamilyMembers', ax=ax[1][1])
sns.countplot(data = data, x = 'ChronicDiseases', ax=ax[1][2])
sns.countplot(data = data, x = 'FrequentFlyer', ax=ax[2][0])
sns.countplot(data = data, x = 'EverTravelledAbroad', ax=ax[2][1])
sns.countplot(data = data, x = 'TravelInsurance', ax=ax[2][2])

sns.set(rc={'figure.figsize':(40,30)})
```


