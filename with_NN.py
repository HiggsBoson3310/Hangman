import numpy as np
import tensorflow as tf
from itertools import combinations

ALPHABET = np.array([ s for s in 'abcdefghijklmnopqrstuvwxyz'])
# determine a fixed order so that we don't repeat words with multiple game instances
sorting = np.random.choice(range(1000),size=1000,replace=False)

class HangManPlayer():
    def __init__(self,nnodes):
        self.create_network(nnodes)

    def create_network(self, nnodes):
        self.hidden_layers = [self.weight_variable(27,nnodes[0])]
        self.hidden_bias = [self.bias_variable(nnodes[0])]
        for ni,nip1 in zip(nnodes[:-1],nnodes[1:]):
            self.hidden_layers.append(self.weight_variable(ni,nip1))
            self.hidden_bias.append(self.bias_variable(nip1))
        self.hidden_layers.append(self.weight_variable(nnodes[-1],26))
        self.hidden_bias.append(self.bias_variable(26))
    
    def modify_layers(self,new_layers, new_bias):
        i = 0
        for layer, bias in zip(new_layers,new_bias):
            self.hidden_layers[i] = layer
            self.hidden_bias[i] = bias
            i += 1

    def get_layers(self):
        return self.hidden_layers, self.hidden_bias  

    def predictions(self,input):
        res = tf.nn.relu(tf.matmul(input,self.hidden_layers[0])+self.hidden_bias[0])
        for l,b in zip(self.hidden_layers[1:],self.hidden_bias[1:]):
            res = tf.nn.relu(tf.matmul(res,l)+b)
        return tf.nn.softmax(res)
        
    def weight_variable(self,dim1, dim2):
        var = tf.random.truncated_normal((dim1,dim2),stddev=0.5,seed=np.random.randint(0,high=250))
        return tf.Variable(var,trainable=True)
    
    def bias_variable(self, dim2):
        var = tf.random.truncated_normal((1,dim2),stddev=0.5,seed=np.random.randint(0,high=250))
        return tf.Variable(var,trainable=True)
    
    def remove_life(self):
        self.lives -= 1

    def initialize_game(self,game):
        self.status = np.zeros(27).astype('float32').reshape(1,-1)
        self.status[0,0] = len(game.word)
        self.lives = 6
        self.won_status = False
    
    def make_a_guess(self,game):
        iin = np.argmax(self.predictions(self.status))
        l_g = ALPHABET[iin]
        res = game.guess(l_g,self)
        self.status[0,iin] = res

    def have_won(self, game):
        self.won_status = game.win_from_status(self)

class Player_populations():
    def __init__(self,size,nnodes):
        self.players = []
        self.nnodes = nnodes
        self.size = size
        for i in range(size):
            self.players.append(HangManPlayer(nnodes))
    
    def initialize_games(self,game):
        for player in self.players:
            player.initialize_game(game)
    
    def overlap_players(self,player1,player2):
        shapes = [(27,self.nnodes[0])]
        
        for i in range(len(self.nnodes)-1):
            shapes.append((self.nnodes[i],self.nnodes[i+1]))
        
        shapes.append((self.nnodes[-1],26))
        parent1_w, parent1_b = player1.get_layers()
        parent2_w, parent2_b = player2.get_layers()

        daughter_w = []
        daughter_b = []

        for i in range(len(shapes)):
            row_or_col = np.random.randint(0,2)
            if(row_or_col==0):
                d1 = np.random.randint(low=0,high=2,size=shapes[i][0])
                d2 = 1-d1
                mask1 = np.diag(d1).astype('float32')
                mask2 = np.diag(d2).astype('float32')
                daughter_w.append(tf.add(tf.matmul(mask1,parent1_w[i]),tf.matmul(mask2,parent2_w[i])))
            else:
                d1 = np.random.randint(low=0,high=2,size=shapes[i][1])
                d2 = 1-d1
                mask1 = np.diag(d1).astype('float32')
                mask2 = np.diag(d2).astype('float32')
                daughter_w.append(tf.add(tf.matmul(parent1_w[i],mask1),tf.matmul(parent2_w[i],mask2)))
            
            d1 = np.random.randint(low=0,high=2,size=shapes[i][1])
            d2 = 1-d1
            mask1 = np.diag(d1).astype('float32')
            mask2 = np.diag(d2).astype('float32')
            daughter_b.append(tf.add(tf.matmul(parent1_b[i],mask1),tf.matmul(parent2_b[i],mask2)))

        return daughter_w, daughter_b
    
    def generation(self,game):
        self.results = []
        for player in self.players:
            player.initialize_game(game)
            while(player.lives>0 and player.won_status==False):
                player.make_a_guess(game)
            self.results.append([player.won_status,len(np.where(player.status>=1)[0])])
        
    
    def purge_and_reproduce(self):
        rep = []
        for i,r in enumerate(self.results):
            if(r[0]): rep.append(i)
        win = rep.copy()
        
        pairs = list(combinations(rep,2))
        if(len(pairs)<self.size):
            tops = np.argsort([a[1] for a in self.results][::-1])
            for t in tops:
                if(t not in rep): rep.append(t)
        pairs = list(combinations(rep,2))
        cnt = 0
        for i in rep:
            if(i not in win):
                dw, db = self.overlap_players(self.players[pairs[cnt][0]],self.players[pairs[cnt][1]])
                self.players[i].modify_layers(dw,db)
                cnt+=1
        
        
class Hangman_Game():
    def __init__(self,word=''):
        if(word==''):
            self.dict = open('./play_set.txt').readlines()
            self.word = np.random.choice(self.dict,size=1)[0][:-1]
        else:
            self.dict = [word]
            self.word = word
        #print(self.word)
        self.word_array = np.array([s for s in self.word])
        self.output = list(('_'*len(self.word))[:])
        self.lives = 6
        self.played_letters = {}
        self.over = False
        self.winner = False
    
    def game_status(self):
        return self.output
    
    def game_over(self):
        if(self.lives <= 0):
            self.over = True
            #print('Game Over')
        elif(''.join(self.output[::])==self.word):
            #print('You won')
            self.winner = True
            self.over = True

    def guess(self,letter):
        if(letter in self.word):
            pos = np.where(self.word_array==letter)[0]
            for i in pos: self.output[i]=letter
        else:
            self.lives -= 1 
        #print('|'*self.lives)
        #print(' '.join(self.output))
        
        self.game_over()
        
        return self.over
        



