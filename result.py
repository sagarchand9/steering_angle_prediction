batch_size_option = [1, 5, 10]
num_epochs_option = [1, 2, 3]
learning_rate_option = [0.001, 0.002, 0.003]
normalisation_option=["Batch","Instance"]
affine_option=["False","True"]
color_space_option=["RGB","YCbCr"]

for a in range(2):
    fila = open('ans.txt','a')
    fila.write('\n\nColor Space'+color_space_option[a]+'\n\n')
    fila.close()		
    for d in range(3):
        fila = open('ans.txt','a')
        fila.write('\n\nLearning rate changed'+'\n\n')
        fila.close()		
    
        for b in range(3):
            for c in range(3):
                for e in range(2):
                    for f in range(2):
                        fi = str(a)+str(b)+str(c)+str(d)+str(e)+str(f)+".txt"
                        with open(fi) as fil:
                            flag=0
                            l = '[1,    20]'
                            train=''
                            for line in fil:
                                if l in line:
                                    if(flag==0):
                                        flag=1
                                    elif(flag==1):
                                        train=temp
									#print(temp)
                                temp=line
							#print line	
						
						
                            l_r=learning_rate_option[d]
                            b_s=batch_size_option[b]
                            n_e=num_epochs_option[c]
                            norm=normalisation_option[e]
                            affine=affine_option[f]
                            tr_e=train[17:]
                            te_e=line[17:]
                            			
                            if(tr_e>'1.0'):
                                tr_e="Bad"
                            if(te_e>'1.0'):
                                te_e="Bad"
                            
                            answer=str(l_r) + ' & ' + str(b_s) + ' & ' + str(n_e) + ' & ' + str(norm) + ' & ' + str(affine) + ' & ' + str(tr_e) + ' & ' + str(te_e) + ' \\\\\\hline' 
						
                            fila = open('ans.txt','a')
                            fila.write(answer)
                            fila.close()		
						#fil = open(fi,'r')
						#message = fil.read()
						
                        fil.close()
