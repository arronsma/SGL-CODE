import os
import io_routine
import random
import math
import sys
import algebra_graph
import machineLearning
import numpy
import argparse
from sklearn.preprocessing import StandardScaler
import gc

os.chdir(os.path.split(os.path.realpath(__file__))[0])
os.chdir('../')
work_file=os.getcwd()
os.chdir('./source')

debug=0
'''no used
pso_w1=0.2
pso_r1=0
pso_c1=0.1
pso_c2=0.2

'''
'''PSO_20
pso_r1=0
pso_w1=0.5
pso_c1=0.17
pso_c2=0.3
'''

pso_r1=0
pso_w1=0.5
pso_c1=0.10
pso_c2=0.10

repeat=10
atom_distances=3
#cx plot in his code
best_N = 10

far_x=10
atom_distances=3

#parameter of AGL
etaE=2
kappa=2.5
etaL=2
v=2

def model_predict_pos(target,rank_iteration,model_list,pop_size,predf='pred'): #this def must start with rank as 0,target must be a str,iteration is a int,
    # model should have been trained,      model_predict_pos('7',10,model,50)
    if not (os.path.exists((work_file+'/'+target+predf))):
        os.mkdir(work_file+'/'+target+predf)
    os.chdir((work_file+'/'+target+predf))
    atom_num=int(target)
    record_file=work_file+'/'+target+predf+'/'+'pred.out'
    pos_all=[]
    ener_all=[]
    gbest_pos=[]
    gbest_ener=[]
    pso_w=pso_w1
    record_all=[]
    pos_all_all=[]
    
    f=io_routine.loadCache(work_file+'/source/0_feature_set_remain_'+target)
    scaler=StandardScaler()
    scaler.fit(f)

    for i5 in range(pop_size):
        pos_all.append([])
        ener_all.append([]) #建立一个 polulatio的列表
        record_all.append([]) #建立一个 polulation*n_atom的列表 [i][j]表示i-1个population的 j-1个原子
        for i_re1 in range(atom_num):
            record_all[i5].append([])
    add_how_much=pop_size
    for i1 in range(10):
        assert(not debug) 
        file_calc=str(i1)
        if os.path.exists(file_calc):
            pass
        else:
            start_file=i1
            break
    for i2 in range(rank_iteration): #开始每代的迭代
        if i2==round(rank_iteration*2/3):
            pso_w=pso_w1/2
        #        if i2<5:
        #            pso_r=0.5
        #        else:
        #            pso_r=0
        file_calc=start_file+i2
        os.mkdir(str(file_calc))#可能是代表第N代，断点续传用，不知道
        posID_all=[]
        pos_itr_all=[]
        vw=[[]]
        for i9 in range(pop_size):
            vw[0].append([])
            for i10 in range(atom_num):
                vw[0][i9].append([0,0,0]) #vw【0】是一个 人口数*原子数 的列表，原子位置为（0,0,0）
        if i2==0 and start_file==0: 
            add_how_much1=round(add_how_much/2)
            add_how_much2=add_how_much-add_how_much1
            pos_itr_all1,posID_all1=random_point_gen(atom_num,[],[],add_how_much1,'BCC')  #atom_num是团簇里的原子，add_how_much1是种群数目的一半，种群由BCC和FCC各一半组成
            pos_itr_all2,posID_all2=random_point_gen(atom_num,[],[],add_how_much2,'FCC')
            posID_all=posID_all1+posID_all2
            pos_itr_all=pos_itr_all1+pos_itr_all2
            for i in range(len(pos_itr_all)):
                pos_all_all.append(pos_itr_all[i]) #pos_all是一代整个种群，pos_all_all是所有代的所有种群,当然，第一个维度还是表示哪个构型，但是第一个维度的长度是 rank*population
                #pos_itr_all[i],e=pos_vibra(pos_itr_all[i],model)

            '''
            for i in range(1,len(pos_itr_all)+1):
                file_calc_1=(str(file_calc)+'/%d'%i)
                os.mkdir(file_calc_1)
                make_vasp(file_calc_1,pos_itr_all[i-1],if_run=False)
                ##predict
            '''
            ''' TFs PSO
            betti_tmp=pos2betti(pos_itr_all)
            far_tmp=pos_farest_dis(pos_itr_all)
            #for j3 in range(len(betti_tmp)):
                #betti_tmp[j3].append(round(far_tmp[j3],2))
                #betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            '''

            '''
            etaE=2
            kappa=2.5
            etaL=2
            v=2.5
            '''
            #AGL特征工程
            featureSet_mul=algebra_graph.pos2feature(pos_itr_all,etaL,kappa,etaE,v)
            featureSet_mul=scaler.transform(featureSet_mul)
            pred_energy=numpy.zeros((pop_size))
            '''
            for i in range(repeat):
                model=io_routine.loadCache(work_file+'/model_cache/model_GBR_'+str(i))
                pred_energy+=model.predict(featureSet_mul)
                print(pred_energy.shape,featureSet_mul.shape)
                del model
                gc.collect()
            '''
            for i in model_list:
                pred_energy+=i.predict(featureSet_mul)
                print(pred_energy.shape,featureSet_mul.shape)               
            pred_energy=pred_energy/repeat
            pred_energy=pred_energy.tolist()
            best_pos_index=pred_energy.index(min(pred_energy))+1
            print(os.getcwd())
            with open("pred.out",'a') as rf:
                rf.write((str(file_calc)+'\n'))
                rf.write(str(best_pos_index)+'\n') 
                rf.write(str(min(pred_energy))+'\n') 
                rf.write(str(pos_itr_all[best_pos_index-1])+'\n') 
            for j3 in range(pop_size):
                ener_all[j3].append(pred_energy[j3])
                pos_all[j3].append(pos_itr_all[j3])
            mag_all=list(range(pop_size)) 
   
            pos2,energy2,mag2=pos_ener_sort(pos_itr_all,pred_energy,mag_all)

            for i in range(best_N):
                gbest_pos.append(pos2[i]) 
                gbest_ener.append(energy2[i])
            '''
            gbest_pos.append(pos2[0]) #选出一次迭代中最好的三个，由于是第一次迭代，所以是全局最优
            gbest_pos.append(pos2[1])
            gbest_pos.append(pos2[2])
            gbest_ener.append(energy2[0])
            gbest_ener.append(energy2[1])
            gbest_ener.append(energy2[2])
            '''

        else:
            if len(gbest_pos)==0:    #first genration are calculated , later are predicted
                pos1,energy1,mag1 = analysis_vasp('%d'%(file_calc-1),atom_num)

            gbest_mag=list(range(len(gbest_pos)))
            gbest_pos2,gbest_ener2,gbest_mag2=pos_ener_sort(gbest_pos,gbest_ener,gbest_mag)
            #gbest_pos,gbest_ener存储了所有已知结构，只要sort就可以找到最优

            gbest_fin_now=gbest_pos[-best_N]   
            gbest_fin_now.sort()
            if len(gbest_pos2)>=4:
                gbest_fin=gbest_pos2[random.randint(0,3)]
            else:
                gbest_fin=gbest_pos2[len(gbest_pos2)-1]
            gbest_fin.sort()
            vw.append([])
            for i3 in range(len(pos_all)):
                pbest_pos=[]
                pbest_ener=[]
                pbest_mag=[]
                vw[-1].append([])
                for i41 in range(len(pos_all[i3])):
                    pbest_pos.append(pos_all[i3][i41])
                    pbest_ener.append(ener_all[i3][i41])
                    pbest_mag.append(1)
                pbest_pos2,pbest_ener2,pbest_mag2=pos_ener_sort(pbest_pos,pbest_ener,pbest_mag)
                pbest_fin=pbest_pos2[0]
#                pbest_fin.sort()
                pos_now=pbest_fin#pbest_pos[-1]
                #pos_now.sort()
                pos_now,gbest_fin_now=search_close_route(pos_now,gbest_fin_now)
                pos_now,gbest_fin=search_close_route(pos_now,gbest_fin)
                go_on1=True
                pso_r=pso_r1
                try_lim=0
                while go_on1:
                    too_far_toomuch=0
                    too_close_toomuch=0
                    new_pos=[]
                    new_pos_w=[]
                    new_pos_c1=[]
                    new_pos_c2=[]
                    
                    for i4 in range(atom_num):

                        #new_pos_w.append(list(map(lambda x: 0*random.randint(-1,1)*random.random() ,pos_now[i4] )) )
                        #z_z=random.randint(-1,1)  
                        new_pos_c1.append(list(map(lambda x,y: (y-x)*pso_c1*random.random(),pos_now[i4],gbest_fin_now[i4] )) )
                        new_pos_c2.append(list(map(lambda x,y: (y-x)*pso_c2*random.random(),pos_now[i4],gbest_fin[i4] )) )
                        v_tmp=list(map(lambda x,y,z: pso_w*x+y+z,vw[-2][i3][i4],new_pos_c1[i4],new_pos_c2[i4]))
                        
                        if random.randint(0,1):
                            v_tmp=list(map(lambda x,y: pso_w*x+y,vw[-2][i3][i4],new_pos_c2[i4]))
                        elif random.randint(0,1):
                            v_tmp=list(map(lambda x,y: pso_w*x+y,vw[-2][i3][i4],new_pos_c1[i4]))
                        else:
                            v_tmp=list(map(lambda x,y,z: pso_w*x+y+z,vw[-2][i3][i4],new_pos_c2[i4],new_pos_c1[i4]))
                        vw[-1][i3].append(list(map(too_quick,v_tmp)))
                        new_pos.append(list(map(lambda y,ori: y+ori+pso_r*random.randint(-1,1)*random.random() ,vw[-1][i3][i4],pos_now[i4])))
                        '''
                        record_all[i3][i4].append(pos_now[i4])
                        record_all[i3][i4].append(pbest_fin[i4])
                        record_all[i3][i4].append(gbest_fin[i4])
                        record_all[i3][i4].append(new_pos_c1[i4])
                        record_all[i3][i4].append(new_pos_c2[i4])
                        '''
                    try_lim=try_lim+1
                    if try_lim== 10000:
                        new_pos=make_pos_fit(new_pos)
                        break
                    if too_far(new_pos,8):
                        print('too far')
                        go_on1=True
                    
                        if pso_r<0.6:
                            pso_r=pso_r+0.02
                
                        too_far_toomuch=too_far_toomuch+1
                        if too_far_toomuch<10:
                            for j_del in range(len(vw[-1][i3])-1,-1,-1):
                                del vw[-1][i3][j_del]
                            continue
                        else:
                            new_pos1,posID_all1=random_point_gen(atom_num,[],[],1,'BCC')
                            new_pos=new_pos1[0]
                    if too_close(new_pos,2.4):
                        print('too close')
                        go_on1=True
                        if pso_r<0.6:
                            pso_r=pso_r+0.02
                            

                        too_close_toomuch=too_close_toomuch+1
                        if too_close_toomuch<10:
                            for j_del in range(len(vw[-1][i3])-1,-1,-1):
                                del vw[-1][i3][j_del]
                            continue
                        else:
                            new_pos1,posID_all1=random_point_gen(atom_num,[],[],1,'BCC')
                            new_pos=new_pos1[0]
                    if check_dup(new_pos,pos_now,(1/far_x)*0.75):
                        go_on1=True
                        for j_del in range(len(vw[-1][i3])-1,-1,-1):
                            del vw[-1][i3][j_del]

                        if pso_r<0.6:
                            pso_r=pso_r+0.02

                        continue
#                    new_pos=make_pos_fit(new_pos)
                    break
                    if check_dup2(pos_all_all,new_pos,(1/far_x)*0.5):
                        go_on1=True
                        for j_del in range(len(vw[-1][i3])-1,-1,-1):
                            del vw[-1][i3][j_del]
                        if pso_r<0.6:
                            pso_r=pso_r+0.02
                        continue
#                    new_pos=make_pos_fit(new_pos)
                    break
                #new_pos,_ = pos2standard_coord([new_pos])
                #new_pos = new_pos[0]
                print(len(new_pos))
                print(len(new_pos[0]))
                pos_all_all.append(new_pos)
#                new_pos,e=pos_vibra(new_pos,model)
                pos_itr_all.append(new_pos)
            '''
            for i7 in range(1,len(pos_itr_all)+1):
                file_calc_1=(str(file_calc)+'/%d'%i7)
                os.mkdir(file_calc_1)
                make_vasp(file_calc_1,pos_itr_all[i7-1],if_run=False)
            '''
            '''
            betti_tmp=pos2betti(pos_itr_all)
            far_tmp=pos_farest_dis(pos_itr_all)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            pred_energy=model.predict(numpy.array(betti_tmp))
            '''
            featureSet_mul=algebra_graph.pos2feature(pos_itr_all,etaL,kappa,etaE,v)
            featureSet_mul=scaler.transform(featureSet_mul)
            pred_energy=numpy.zeros((pop_size))
            '''
            for i in range(repeat):
                model=io_routine.loadCache('../model_cache/model_GBR_'+str(i))
                pred_energy+=model.predict(featureSet_mul)
                del model
                gc.collect()
            '''
            for i in model_list:
                pred_energy+=i.predict(featureSet_mul)
                print(pred_energy.shape,featureSet_mul.shape)  
            pred_energy=pred_energy/repeat
            pred_energy=pred_energy.tolist()
            best_pos_index=pred_energy.index(min(pred_energy))+1

            with open(record_file,'a') as rf:
                rf.write((str(file_calc)+'\n'))
                rf.write(str(best_pos_index)+'\n')  
                rf.write(str(min(pred_energy))+'\n') 
                rf.write(str(pos_itr_all[best_pos_index-1])+'\n') 
            for j3 in range(pop_size):
                ener_all[j3].append(pred_energy[j3])
                pos_all[j3].append(pos_itr_all[j3])
            mag_all=list(range(pop_size))  
            pos_21,energy_21,mag_21=pos_ener_sort(pos_itr_all,pred_energy,mag_all)
            '''
            gbest_pos.append(pos_21[0])
            gbest_pos.append(pos_21[1])
            gbest_pos.append(pos_21[2])
            gbest_ener.append(energy_21[0])
            gbest_ener.append(energy_21[1])
            gbest_ener.append(energy_21[2])
            '''
            for i in range(best_N):
                gbest_pos.append(pos_21[i])
                gbest_ener.append(energy_21[i])
        os.chdir((work_file+'/'+target+predf))
    gbest_mag=list(range(len(gbest_pos)))
    gbest_pos2,gbest_ener2,gbest_mag2=pos_ener_sort(gbest_pos,gbest_ener,gbest_mag)
    already_cal=[]


    for best_i in range(1,len(gbest_pos2)):#len(gbest_pos2)+1
        pwd_file=os.getcwd()
        if check_dup2(already_cal,gbest_pos2[best_i-1],4/far_x):
            continue
        already_cal.append(gbest_pos2[best_i-1])
        file_calc_best=pwd_file+'/final'+str(best_i)
        os.mkdir(file_calc_best)
        try:
            pass
            #make_vasp(file_calc_best,gbest_pos2[best_i-1],False,True,if_pred=True)
        except:
            print("vasp failed")
        os.chdir(pwd_file)
    print("caching data")
    io_routine.cacheByJobLib(gbest_pos2,"gbest_pos2_"+target+'_'+str(rank_iteration))
    io_routine.cacheByJobLib(gbest_ener2,"gbest_ener2"+target+'_'+str(rank_iteration))
    io_routine.cacheByJobLib(gbest_pos,"gbest_pos"+target+'_'+str(rank_iteration))
    io_routine.cacheByJobLib(gbest_ener,"gbest_ener"+target+'_'+str(rank_iteration))
    io_routine.cacheByJobLib(already_cal,"already"+target+'_'+str(rank_iteration))
#        with open('final%d.gjf'%(best_i-1),'w') as fg:
#            fg.write('%chk=C.chk\n%mem=10GB\n\n# bpw91/6-31G pop=full nosym scf=(maxcycle=80,xqc)\n\npso\n\n')
#            fg.write('0 %d\n'%gbest_mag2[best_i-1])
#            for item in gbest_pos2[best_i]:
#                fg.write('Li %15.8f %15.8f %15.8f\n' % tuple(item))
#            fg.write('\n')  
    return already_cal

def random_point_gen(atom_num,pos_all,posID_all,add_how_much,label):#atom_num是团簇里的原子，add_how_much是种群数目，pos_all是带进来的种群，要把pos_all补充到add_how_much
    #print('random_point_gen begin')
    if label == 'BCC':
        print(os.getcwd())
        lattice,pos=readposcar('POSCAR_BCC') #读取PSOCAR得到晶格和原子位置 DIRECT的
    if label == 'FCC':
        lattice,pos=readposcar('POSCAR_FCC')
    if atom_num<10:
        N_super=round(atom_num*3)
    else:
        N_super=round(atom_num)
    supercell,superID=getsuperpoint(lattice,pos,N_super) #扩胞N*N*N，supercell是每个原子的位置，superID是每个原子对应的坐标
    center_atom=round((N_super-1)/2)#找到中心的晶胞
    origi_posall_len=len(pos_all)
    while len(pos_all)-origi_posall_len<add_how_much: 
        pos=[]
        pos_real=[]
        while len(pos)<atom_num:
            if len(pos)==0:#第一个原子取中心
                pos.append([center_atom,center_atom,center_atom,0])#pos是原子在superID中的索引
                pos_real.append(supercell[superID[center_atom][center_atom][center_atom][0]])#superID[center_atom][center_atom][center_atom][0]是在supercell里面的索引
            else:
                len_pos=len(pos)
                mom_atom=random.randint(0,(len_pos-1))
                temp=search_close_point(pos[mom_atom],pos,N_super,label) 
                if temp==[]:
                    continue
                temp_real=supercell[superID[temp[0]][temp[1]][temp[2]][temp[3]]]
                #print(len(pos_all))
                #print(len_pos)
                pos.append(temp)
                pos_real.append(temp_real)#pos保存一个构型20个原子的四维索引，pos_real保存一个构型20个原子的3维坐标
        add_pos=True
        for i in range(len(pos_all)):

            if check_dup(pos_all[i],pos_real):
                add_pos=False
        if add_pos:
            
            posID_all.append(pos)
#            pos_real=make_pos_fit(pos_real)
            pos_all.append(pos_real)
    pos_all,cc_tmp=pos2standard_coord(pos_all)
    #print('random_point_gen done')
    return pos_all,posID_all #pso_all比pos维度多1，保存整个种群所有的构型 pos_all[0]是第一个构型 pos_all[0][0]是第一个构型第一个原子三维坐标

def readposcar(target_pos): #read file poscar, get lattice and atom_pos
    pos = []
    print(os.getcwd())
    f = open(target_pos)
    try:
        for line in f:
            pos.append(line)
    except:
        f.close()
    lattice = []
    pos_all=[]
    for item in pos[2:5]:
        try:
            lattice.append(list(map(float, item.split())))
        except:
            return False
    for item in lattice:
        if len(item) != 3: return False
    #for item in pos[6]:
     #   try:
      #      orig_atom_num=list(map(float, item.split()))
       # except:
        #    return False 
    for item in pos[8:]:
        try:
            pos_all.append(list(map(float, item.split())))
        except:
            return False 
    return([lattice,pos_all])

def getsuperpoint(lattice,pos,N): #N是团簇里面原子的数目
    ID=0
    superpoint=[]
    superID=[]
    for xi in range(N):
        superID.append([])
        for yi in range(N):
            superID[xi].append([])
            for zi in range(N):
                superID[xi][yi].append([])
                for posi in range(len(pos)):
                    tempx=pos[posi][0]*lattice[0][0]+pos[posi][1]*lattice[1][0]+pos[posi][2]*lattice[2][0]+xi*lattice[0][0]+yi*lattice[1][0]+zi*lattice[2][0]
                    tempy=pos[posi][0]*lattice[0][1]+pos[posi][1]*lattice[1][1]+pos[posi][2]*lattice[2][1]+xi*lattice[0][1]+yi*lattice[1][1]+zi*lattice[2][1]
                    tempz=pos[posi][0]*lattice[0][2]+pos[posi][1]*lattice[1][2]+pos[posi][2]*lattice[2][2]+xi*lattice[0][2]+yi*lattice[1][2]+zi*lattice[2][2] #转换为笛卡尔坐标，原子位置矩阵放在左边，晶格矢量放右边
                    superpoint.append([tempx,tempy,tempz])
                    superID[xi][yi][zi].append(ID)
                    ID=ID+1
    return superpoint,superID #N*N*N*晶胞内的原子

def search_close_point(ID,exit_ID,N_super,label='FCC'):    #!only for FCC, and 0 is z_low atom, 1 is z_high atom!  #only for FCC  #only for FCC  #only for FCC  #only for FCC  #only for FCC 
    #print('search_close_point begin') ID是中心原子的四级索引
    if label == 'FCC':
        close_ID1=[ [ID[0]+1,ID[1],ID[2],ID[3]] , [ID[0],ID[1]+1,ID[2],ID[3]] , [ID[0]+1,ID[1]+1,ID[2],ID[3]] , [ID[0]-1,ID[1],ID[2],ID[3]] , [ID[0],ID[1]-1,ID[2],ID[3]], [ID[0]-1,ID[1]-1,ID[2],ID[3]] ]
        if ID[3]==0:
            close_ID2=[ [ID[0],ID[1],ID[2],ID[3]+1] , [ID[0],ID[1]+1,ID[2],ID[3]+1] , [ID[0]-1,ID[1],ID[2],ID[3]+1] ]
            close_ID3=[ [ID[0],ID[1],ID[2]-1,ID[3]+1] , [ID[0],ID[1]+1,ID[2]-1,ID[3]+1] , [ID[0]-1,ID[1],ID[2]-1,ID[3]+1] ]
        if ID[3]==1:
            close_ID2=[ [ID[0],ID[1],ID[2],ID[3]-1] , [ID[0],ID[1]-1,ID[2],ID[3]-1] , [ID[0]+1,ID[1],ID[2],ID[3]-1] ]
            close_ID3=[ [ID[0],ID[1],ID[2]+1,ID[3]-1] , [ID[0],ID[1]-1,ID[2]+1,ID[3]-1] , [ID[0]+1,ID[1],ID[2]+1,ID[3]-1] ]
        close_ID=close_ID1 + close_ID2 + close_ID3
        for i in range(len(close_ID)):
            close_ID[i]=[close_ID[i][0]%N_super,close_ID[i][1]%N_super,close_ID[i][2]%N_super,close_ID[i][3]]
    if label == 'BCC':
        if  ID[3]==0:
            close_ID1=[ [ID[0],ID[1],ID[2],1] , [ID[0]-1,ID[1],ID[2],1] , [ID[0],ID[1]-1,ID[2],1] , [ID[0]-1,ID[1]-1,ID[2],1] ]
            close_ID2=[ [ID[0],ID[1],ID[2]-1,1] , [ID[0]-1,ID[1],ID[2]-1,1] , [ID[0],ID[1]-1,ID[2]-1,1] , [ID[0]-1,ID[1]-1,ID[2]-1,1] ]
        if  ID[3]==1:
            close_ID1=[ [ID[0],ID[1],ID[2],0] , [ID[0]+1,ID[1],ID[2],0] , [ID[0],ID[1]+1,ID[2],0] , [ID[0]+1,ID[1]+1,ID[2],0] ]
            close_ID2=[ [ID[0],ID[1],ID[2]+1,0] , [ID[0]+1,ID[1],ID[2]+1,0] , [ID[0],ID[1]+1,ID[2]+1,0] , [ID[0]+1,ID[1]+1,ID[2]+1,0] ]
        close_ID=close_ID1 + close_ID2 
        for i in range(len(close_ID)):
            close_ID[i]=[close_ID[i][0]%N_super,close_ID[i][1]%N_super,close_ID[i][2]%N_super,close_ID[i][3]]
    x_try=0
    while x_try<7:
        x_try=x_try+1
        if label == 'FCC':
            No_ID_temp=random.randint(0,11)      #!  first_close atom number
        if label == 'BCC':
            No_ID_temp=random.randint(0,7)      #!  first_close atom number            
        if close_ID[No_ID_temp] not in exit_ID:
            #print('search_close_point done')
            return close_ID[No_ID_temp]
    return []

def check_dup(pos1,pos2,limit=0.1):
#    pos1_tmp,c1_pmat=pos2standard_coord([pos1])
#    pos2_tmp,c2_pmat=pos2standard_coord([pos2])
#    c1_mat=pos_center_dist_mat(pos1_tmp[0],[0,0,0])
#    c2_mat=pos_center_dist_mat(pos2_tmp[0],[0,0,0])
#    c1_mat.sort()
#    c2_mat.sort()
    pos1_mat=dist_matric(pos1)   
    pos2_mat=dist_matric(pos2) 
    for def_i2 in range(len(pos1_mat)):
        if abs(abs(pos1_mat[def_i2]) - abs(pos2_mat[def_i2]))>limit:
            return False
    return True
def check_dup2(pos1_list,pos2,limit=0.1):
    for item1 in range(len(pos1_list)):
        if check_dup(pos1_list[item1],pos2,limit):
            return True
    return False
def too_quick(v):
    if abs(v)>0.5*atom_distances:
        return v/abs(v)*0.5*atom_distances
    else:
        return v
def too_close(pos,dis=False):
    min_dis=math.pow(atom_distances-1.5,2)#-0.8
    if dis:
        min_dis=math.pow(dis,2)
    for i in range(1,len(pos)):
        for j in range(i):
            if  math.pow(pos[i][0]-pos[j][0],2)+math.pow(pos[i][1]-pos[j][1],2)+math.pow(pos[i][2]-pos[j][2],2) <min_dis:
                #print(math.pow(pos[i][0]-pos[j][0],2)+math.pow(pos[i][1]-pos[j][1],2)+math.pow(pos[i][2]-pos[j][2],2))
                return True
    return False
def too_far(pos,dis=False):
    max_dis=math.pow(atom_distances+1,2)
    if dis:
        max_dis=math.pow(dis,2)
    tmp_list=[pos[0]]
    for k in range(len(pos)):
        for i in tmp_list:
            for j in pos:
                if  math.pow(i[0]-j[0],2)+math.pow(i[1]-j[1],2)+math.pow(i[2]-j[2],2) < max_dis and j not in tmp_list:#+0.3
                    tmp_list.append(j)
    if len(tmp_list)!=len(pos):

        return True
    else:
        return False

def dist_matric(pos): 
    # input: pos[n,3]matrix,  each row is the  coordinate of an atom
    # output: dis_mat[1,n]matrix, each elemnt is the distance between each 2 atoms ordered from samll to large, (you don't know for example dis_mat[2] is the distance between which two atoms.)
    dis_mat=[]
    if len(pos)<100:
        for def_i in range(1,len(pos)):
            for def_j in range(def_i):
                temp=math.sqrt(math.pow(pos[def_j][0]-pos[def_i][0],2)+math.pow(pos[def_j][1]-pos[def_i][1],2)+math.pow(pos[def_j][2]-pos[def_i][2],2))
                dis_mat.append(temp)
    dis_mat.sort()
    return(dis_mat)

def pos2standard_coord(pos):  #pos list should be assgined here,we choose mass center point as center point , and the longgert distance 
    new_pos=[]                            # as x axis 选取一个中心进行坐标变换
    for i_1 in range(len(pos)):
        pos_tmp=[]
        x_sum=0
        y_sum=0
        z_sum=0
        for j_1 in range(len(pos[i_1])):
            x_sum=x_sum+pos[i_1][j_1][0]
            y_sum=y_sum+pos[i_1][j_1][1]
            z_sum=z_sum+pos[i_1][j_1][2]
        center=[x_sum/len(pos[i_1]),y_sum/len(pos[i_1]),z_sum/len(pos[i_1])]
        for j_1 in range(len(pos[i_1])):
            pos_tmp.append([pos[i_1][j_1][0]-center[0] , pos[i_1][j_1][1]-center[1] , pos[i_1][j_1][2]-center[2]])
        new_pos.append(pos_tmp)
    return new_pos,center


def pos_ener_sort(pos,energy,mag): #按照energy进行排序，同时保持energy和mag的顺序,从小到大排列
    if len(pos)==len(energy):
        item_num=len(energy)
        for i1 in range(item_num):
            for i2 in range(1,item_num):
                if energy[i2-1]>energy[i2]:
                    energy_tmp=energy[i2-1]
                    energy[i2-1]=energy[i2]
                    energy[i2]=energy_tmp
                    pos_tmp=pos[i2-1]
                    pos[i2-1]=pos[i2]
                    pos[i2]=pos_tmp
                    mag_tmp=mag[i2-1]
                    mag[i2-1]=mag[i2]
                    mag[i2]=mag_tmp                
    else:
        print('no equal')
    return pos,energy,mag

def search_close_route(pos1,pos2):
    pos_tmp=numpy.array(pos2)
    pos2=pos_tmp.tolist()
    pos_tmp=numpy.array(pos1)
    pos1=pos_tmp.tolist()
    #pos1.sort()
    pos2.sort()
    new_pos2=[]
    #print(pos1)
    #print(pos2)
    for ii_1 in range(len(pos1)):#遍历pos1的每个原子
        dist_mat=[]
        for ii_2 in range(len(pos2)):#遍历pos2的每个原子
            temp=math.pow(pos1[ii_1][0]-pos2[ii_2][0],2)+math.pow(pos1[ii_1][1]-pos2[ii_2][1],2)+math.pow(pos1[ii_1][2]-pos2[ii_2][2],2)
            dist_mat.append(temp)
        no_temp_close=dist_mat.index(min(dist_mat))
        new_pos2.append(pos2[no_temp_close]) #找到pos2中和pos1对应原子距离最近的原子
        del pos2[no_temp_close]#删除pos2的原子，防止重复寻找，由于pos1的原子也是顺序遍历的，所以不会重复
    return pos1,new_pos2

def make_pos_fit(pos):
    pos2,c1=pos2standard_coord([pos])
    pos1=pos2[0]
    max_dis0=math.pow(atom_distances-0.2,2)
    max_dis1=math.pow(atom_distances+0.4,2)
    max_dis2=math.pow(atom_distances+0.8,2)
    iter_1=0
#    for iter_1 in range(len(pos1)*2):
    while True:
        iter_1= iter_1+1
        change_sum=0
        for i_11 in range(len(pos1)-1):
            i_1=random.randint(0,len(pos1)-2)
            for j_1 in range(i_1+1,len(pos1)):
                bef_dis=math.pow(pos1[i_1][0]-pos1[j_1][0],2) + math.pow(pos1[i_1][1]-pos1[j_1][1],2) +math.pow(pos1[i_1][2]-pos1[j_1][2],2)
                if bef_dis > max_dis0 and bef_dis < max_dis1+0.1 :
                    ltr_dis=math.sqrt(bef_dis/math.pow(atom_distances,2))
                    bef_x=pos1[j_1][0]-pos1[i_1][0]
                    bef_y=pos1[j_1][1]-pos1[i_1][1]
                    bef_z=pos1[j_1][2]-pos1[i_1][2]
                    ltr_x=(bef_x/ltr_dis+bef_x*3)/4
                    ltr_y=(bef_y/ltr_dis+bef_y*3)/4
                    ltr_z=(bef_z/ltr_dis+bef_z*3)/4
                    change_sum=change_sum+abs(ltr_x-bef_x)+abs(ltr_y-bef_y)+abs(ltr_y-bef_y)
                    pos1[j_1]=[pos1[i_1][0]+ltr_x,pos1[i_1][1]+ltr_y,pos1[i_1][2]+ltr_z]
                           
                elif bef_dis > max_dis1 and bef_dis < max_dis2 :
                    ltr_dis=math.sqrt(bef_dis/math.pow(atom_distances+0.4,2))
                    bef_x=pos1[j_1][0]-pos1[i_1][0]
                    bef_y=pos1[j_1][1]-pos1[i_1][1]
                    bef_z=pos1[j_1][2]-pos1[i_1][2]
                    ltr_x=(bef_x/ltr_dis+bef_x*3)/4
                    ltr_y=(bef_y/ltr_dis+bef_y*3)/4
                    ltr_z=(bef_z/ltr_dis+bef_z*3)/4
                    pos1[j_1]=[pos1[i_1][0]+ltr_x,pos1[i_1][1]+ltr_y,pos1[i_1][2]+ltr_z]
                    change_sum=change_sum+abs(ltr_x-bef_x)+abs(ltr_y-bef_y)+abs(ltr_y-bef_y)
                elif bef_dis < max_dis0  :
                    ltr_dis=math.sqrt(bef_dis/math.pow(atom_distances-0.3,2))
                    bef_x=pos1[j_1][0]-pos1[i_1][0]
                    bef_y=pos1[j_1][1]-pos1[i_1][1]
                    bef_z=pos1[j_1][2]-pos1[i_1][2]
                    ltr_x=(bef_x/ltr_dis+bef_x*3)/4
                    ltr_y=(bef_y/ltr_dis+bef_y*3)/4
                    ltr_z=(bef_z/ltr_dis+bef_z*3)/4
                    pos1[j_1]=[pos1[i_1][0]+ltr_x,pos1[i_1][1]+ltr_y,pos1[i_1][2]+ltr_z]
                    change_sum=change_sum+abs(ltr_x-bef_x)+abs(ltr_y-bef_y)+abs(ltr_y-bef_y)
        if change_sum<0.005 or iter_1>50:
            break                    
    return pos1

def make_vasp(target,point,if_run,if_opt=True,if_pred=False):
    file_Kpt=target+'/KPOINT'
    file_INC=target+'/INCAR'
    file_POS=target+'/POSCAR'
    with open( file_Kpt,'w' ) as fk:
        fk.write('AUTO GRID\n0\nG\n1 1 1\n0 0 0')
    with open( file_INC,'w' ) as fi:
        fi.write('SYSTEM=PSO\nLWAVE=.FALSE.\nLCHARG=.FALSE.\nISTART=0\nALGO=Fast\nENCUT=500\nLORBIT=11\nISMEAR=0\nPREC=Normal\n')
        fi.write('NELMIN=3\nPOTIM=0.2\nISIF=2\nEDIFF=0.1E-3\nEDIFFG=-1.0E-2\nLREAL = .False.\nNCORE= 4\nSIGMA=0.1\n')
        fi.write('NELM=150\n')
        if if_opt and not if_pred:
            fi.write('NSW=20\nIBRION=2\nISPIN=1\n')
        if if_opt and if_pred:
            fi.write('NSW=500\nIBRION=3\nISPIN=2\n')
        else:
            fi.write('NSW=0\n')
    io_routine.point2poscar(file_POS,point)
    os.system(r'cp ../submitVasp.pbs %s'%target)
    os.system('cp ../POTCAR %s'%target)
    if if_run:
        orig_os=os.getcwd()
        print(orig_os)
        #os.chdir(target)
        os.system('qsub submitVasp.pbs')
        #os.chdir(orig_os)

def runBatchVasp():
    work_dir = '/share/home/mashm/code/new_AGL/PSO/20pred/'
    a = io_routine.loadCache(work_dir + 'already20_15')
    for i in range(10):
        os.mkdir(work_dir + "run" + str(i))
        os.chdir(work_dir + "run" + str(i))
        make_vasp(work_dir + "run" + str(i), a[i], True,True,True)
        os.chdir("../")

'''
if __name__=="__main__":
    model=io_routine.loadCache("../model_cache/model_0")
    model_predict_pos("40",15,model,2000)
'''
def parse_args(args):
    parser=argparse.ArgumentParser(description="generate command")
    parser.add_argument('--atomNum',type=str,help='Lin',default='20')
    parser.add_argument('--generation',type=int,help='generation of PSO',default=15)
    parser.add_argument('--modelDir',type=str,help='dir for ML model',default='../model_cache')
    parser.add_argument('--population',type=int,help='population',default=40)
    args=parser.parse_args()
    return args

def main(args):
    model_list=[]
    for i in range(repeat):
        model=io_routine.loadCache(work_file+'/model_cache/model_GBR_'+str(i))
        model_list.append(model)
        print(len(model_list))
    model_predict_pos(args.atomNum,args.generation,model_list,args.population,predf='pred')


def cli_main():
    args=parse_args(sys.argv[1:])
    print(args)
    main(args)

if __name__=="__main__":
    
    cli_main()
    #runBatchVasp()
    print('End!')