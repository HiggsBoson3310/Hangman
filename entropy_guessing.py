import pandas as pd
import numpy as np
from itertools import combinations
import re

def surprise(x):
    if(x!=0): return -1*x*np.log2(x)
    else: return 0

def is_pattern(sep_word, let, pattern):
    pos = ''.join(['0' if aa!=let else '1' for aa in sep_word])
    return pos==pattern

def get_pattern(sep_word, let):
    pos = ''.join(['0' if aa!=let else '1' for aa in sep_word])
    return pos

def guess_entropy(self,word ,game):
        wo = word.split(' ')[:-1] #game.game_status()
        print(wo)
        L = len(wo)
        if(len(self.guessed_letters)>=1):
            print(self.w2)
            let = self.guessed_letters[-1]
            self.found_patterns[let] = self.get_pattern(wo,let)
            if(len(self.w2)!=0):
                self.w2 = self.w2[[self.is_pattern(aa,let,self.found_patterns[let]) for aa in self.w2['word_sep']]]
        else:
            self.w2 = self.data[self.data['length']==L]
            self.available_letters = self.engletters
            self.found_patterns = {}
        
        if(len(self.w2)>4):

            #base = list('0'*L)
            
            probabilities = {}

            for i in range(L+1):
                combs = combinations(range(L),i)
                for c in list(combs):
                    b = ''.join(['1' if j in c else '0' for j in range(L)])
                    probabilities[b] = 0
            PT = []
            H = []
            cnt = 0
            for l in self.available_letters:
                PT.append(probabilities.copy())
                for a in self.w2['word_sep']:
                    pos = ''.join(['0' if aa!=l else '1' for aa in a])
                    PT[cnt][pos]+=1/len(self.w2)
                H.append(sum(self.surprise(PT[cnt][x]) for x in PT[cnt]))
                cnt += 1

            let = self.available_letters[np.argmax(H)]
            self.available_letters = np.delete(self.available_letters,np.where(self.available_letters==let)[0])
        elif(len(self.w2)==1):
            print('single word')
            ws = list(self.w2['word_sep'])[0]
            poss = [t for t in ws if t in self.available_letters]
            print(poss)
            let = poss[0]
            self.available_letters = np.delete(self.available_letters,np.where(self.available_letters==let)[0])
        else:
            new_status = word.split(' ')[:-1]
            let_pos = [i for i in range(len(new_status)) if new_status[i]!='_']
            limits = []
            if(not(0 in let_pos)):
                limits.append(['^'+''.join(new_status[0:let_pos[0]+1]),0,new_status[0:let_pos[0]+1]])
            for l in range(len(let_pos)-1):
                if(let_pos[l+1]-let_pos[l] > 1): 
                    if(let_pos[l]==0): 
                        limits.append(['^'+(''.join(new_status[let_pos[l]:let_pos[l+1]+1])).replace('_','.'),let_pos[l],new_status[let_pos[l]:let_pos[l+1]+1]])
                    else: 
                        limits.append(['\B'+''.join(new_status[let_pos[l]:let_pos[l+1]+1]),let_pos[l],new_status[let_pos[l]:let_pos[l+1]+1]])
                    
                    if(let_pos[l+1]==len(new_status)-1): limits[-1][0] += '$'
                    else: limits[-1][0] += '\B'
            if(not(len(new_status)-1 in let_pos)):
                limits.append(['\B'+(''.join(new_status[let_pos[-1]:len(new_status)]).replace('_','.'))+'$',let_pos[-1],new_status[let_pos[-1]:]])
            
            
            prob_let = dict([(a,0) for a in self.available_letters])
            for r,l,sep in limits:
                indices  = [True if(re.match(r,w)) else False for w in self.data['words']]
                w2_new = self.data[indices]
                pos = [i for i in range(len(sep)) if sep[i]=='_']
                if(len(w2_new)!=0):
                    for k,p in enumerate(pos):
                        for a in self.available_letters:
                            prob_let[a] += sum([aa[p+l]==a for aa in w2_new['word_sep']])/(len(w2_new))
                        
            if(not(all([prob_let[a] for a in self.available_letters])==0)):
                let = self.available_letters[np.argmax([prob_let[aa] for aa in self.available_letters])]
                self.available_letters = np.delete(self.available_letters,np.where(self.available_letters==let)[0])
            else:
                # Divide and try to solve each part based on roots or prefixes and suffixes
                sofar1, sofar2 = wo[:L//2], wo[L//2:]
                l1 = len([i for i in range(len(sofar1)) if sofar1[i]=='_'])
                l2 = len([i for i in range(len(sofar2)) if sofar2[i]=='_'])
                coin = np.random.random()
                if(l1>l2):
                    indices  = [True if(re.match('^'+"".join(sofar1).replace('_','.')+'\w+',w)) else False for w in self.data['words']]
                    w2_new = self.data[indices]
                    pos = [i for i in range(len(sofar1)) if sofar1[i]=='_']
                    l = 0
                    
                elif(l2>l1):
                    indices = [True if(re.match('\w+'+"".join(sofar2).replace('_','.')+'$',w)) else False for w in self.data['words']]
                    w2_new = self.data[indices]
                    pos = [i for i in range(len(sofar2)) if sofar2[i]=='_']
                    l = len(sofar2)
                else:
                    if(coin>0.5):
                        indices  = [True if(re.match('^'+"".join(sofar1).replace('_','.')+'\w+',w)) else False for w in self.data['words']]
                        w2_new = self.data[indices]
                        pos = [i for i in range(len(sofar1)) if sofar1[i]=='_']
                        l = 0
                        #print('root')
                    else:
                        indices = [True if(re.match('\w+'+"".join(sofar2).replace('_','.')+'$',w)) else False for w in self.data['words']]
                        w2_new = self.data[indices]
                        pos = [i for i in range(len(sofar2)) if sofar2[i]=='_']
                        l = len(sofar2)
                        #print('suff')

                    
                if(len(w2_new)!=0):
                    prob_let = dict([(a,0) for a in self.available_letters])
                    for k,p in enumerate(pos):
                        for a in self.available_letters:
                            prob_let[a] += sum([aa[p-l]==a for aa in w2_new['word_sep']])/(len(w2_new))
                        
                    let = self.available_letters[np.argmax([prob_let[aa] for aa in self.available_letters])]
                    self.available_letters = np.delete(self.available_letters,np.where(self.available_letters==let)[0])
                
                else:
                    pos = [i for i in range(len(game.output)) if game.output[i]=='_']
                    max_prob = np.empty((len(pos),2),dtype=object)
                    for k,p in enumerate(pos):
                        prob_let = {}
                        
                        for a in self.available_letters:
                            prob_let[a] = sum([aa[p]==a for aa in self.w2_total['word_sep']])/(len(self.w2_total))
                        max_prob[k] = [self.available_letters[np.argmax([prob_let[aa] for aa in self.available_letters])],
                                    max([prob_let[aa] for aa in self.available_letters])]
                    
                    let = max_prob[np.argmax(max_prob[:,1]),0]
                    self.available_letters = np.delete(self.available_letters,np.where(self.available_letters==let)[0])

        return let



# What if we try the output in terms of diatnce between letters
            
            
            