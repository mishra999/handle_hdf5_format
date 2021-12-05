# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:31:46 2021

@author: mmishra
"""
#https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

import json
import h5py
import time
import os

time_samples = np.zeros((4,1024,1024))#channel,trigger cell, timebins

with h5py.File(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult\8to1_mult.hdf5', 'r') as f:
    data_set1 = f['8to1_data/8to1mult']#just pointer to data (data not yet loaded to memory)
    data_set2 = f['8to1_data/time_cells']
    data_set3 = f['8to1_data/time_bins']
    print(len(data_set1))
#   f.close()
   
    time_bins = data_set3[()]
    #get time arrays for all trigger cells
    for i in range(4):#channles
        for j in range(1024): #trigger cell
            temptime = np.zeros(1024)
            for k in range(1024): #timebins
                q, r = divmod(j+k,1024)
                if q:
                    temptime[k] = np.sum(time_bins[0,i,j:(j+k)]) + np.sum(time_bins[0,i,0:r])
                else:
                    temptime[k] = np.sum(time_bins[0,i,j:(j+k)])
            
            time_samples[i,j] = np.copy(temptime)
            
    records = data_set1
    tcell = data_set2[()]
    plt.plot(records[1923, 0])
    plt.show()
#    f.close()
    
    a00 = np.zeros((len(data_set1), 4, 1000))
    for i in range(int(len(data_set1)/10000)):
        ct = 0
        record10000 = records[i*10000:(i+1)*10000,:,:] 
        tcell10000 = tcell[i*10000:(i+1)*10000,:]
        for j in range(10000):
            for i1 in range(4):
                y1 = np.longdouble(record10000[j,i1]) - np.mean(np.longdouble(record10000[j,i1,5:80])) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                x1 = time_samples[i1,int(tcell10000[j,0])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                
                #linear interp
                xs = np.arange(15,1015,1.)#160/4000 = 0.04
                f2 = interp1d(x1,y1) #,kind='previous'
                a00[i*10000+j,i1,:] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]
        print(i)
#    f1.close()
#    f.close()

plt.plot(a00[1923, 0])

def get_data(path, name, dataset_names):
    time_samples = np.zeros((4,1024,1024))

    with h5py.File(path  +  '\\' + name, 'r') as f:
        data_set1 = f[dataset_names[0]]#just pointer to data (data not yet loaded to memory)
        data_set2 = f[dataset_names[1]]
        data_set3 = f[dataset_names[2]]
        print(len(data_set1))
    #   f.close()
       
        time_bins = data_set3[()]
        #get time arrays for all trigger cells
        for i in range(4):#channles
            for j in range(1024): #trigger cell
                temptime = np.zeros(1024)
                for k in range(1024): #timebins
                    q, r = divmod(j+k,1024)
                    if q:
                        temptime[k] = np.sum(time_bins[0,i,j:(j+k)]) + np.sum(time_bins[0,i,0:r])
                    else:
                        temptime[k] = np.sum(time_bins[0,i,j:(j+k)])
                
                time_samples[i,j] = np.copy(temptime)
                
        records = data_set1
        tcell = data_set2[()]
        
        a00 = np.zeros((len(data_set1), 4, 1000))
        for i in range(int(len(data_set1)/10000)):
            ct = 0
            record10000 = records[i*10000:(i+1)*10000,:,:] 
            tcell10000 = tcell[i*10000:(i+1)*10000,:]
            for j in range(10000):
                for i1 in range(4):
                    y1 = np.longdouble(record10000[j,i1]) - np.mean(np.longdouble(record10000[j,i1,5:80])) #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                    x1 = time_samples[i1,int(tcell10000[j,0])] #,np.argmax(vch1[0,i])-50:np.argmax(vch1[0,i])+70
                    
                    #linear interp
                    xs = np.arange(15,1015,1.)#160/4000 = 0.04
                    f2 = interp1d(x1,y1) #,kind='previous'
                    a00[i*10000+j,i1,:] = f2(xs)#f2(xs)#- np.mean(f2(xs)),y1[15:1015]
            print(i)
    return a00

path = r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult'
name = '8to1_mult.hdf5'
dataset_names = ['8to1_data/8to1mult', '8to1_data/time_cells', '8to1_data/time_bins']

drs_pulses = get_data(path, name, dataset_names)

def get_imp_response(path, name, key):
    qq=drs_fdm_parser(path + '\\' + name,key,1,2,23000)
    h_t, cor22, cor11 = qq.imp_response()
    return h_t[0:1000]

path_imp_response = r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult\new'
name_imp_response = {'50':'impres50.dat', '55':'impres55.dat', '70':'impres70.dat'}

imp_response ={'50':np.zeros(1000), '55':np.zeros(1000), '70':np.zeros(1000)} 
for key, value in imp_response.items():
    print(key)
    imp_response[key] = get_imp_response(path_imp_response, name_imp_response[key], key)
    plt.plot(np.abs(np.fft.fft(imp_response[key])))



for value in imp_response.values():    
    plt.plot(np.abs(np.fft.fft(value)))
    



import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((240/500),(290/500) , 3, 20) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
b, a = signal.butter(N, Wn, 'low')


#check filter
#50 MHz

      
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.interpolate import UnivariateSpline
N, Wn = signal.buttord((200/500),(250/500) , 3, 10) #(240/500),(290/500) , 3, 10, charge        (130/500),(170/500) , 3, 10, timing
b, a = signal.butter(N, Wn, 'low')

#h50_f = np.fft.fft(h50_t1[0:1050])
#rec_50 = np.zeros((len(res_50),1050))
#for i in range(len(res_50)):
#    out_x =  np.fft.fft(np.lib.pad(res_50[i,0], (0,50), 'constant', constant_values=(0., 0.)))/h50_f
#    out_xn = np.real(np.fft.ifft(out_x))
#    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn)
    
h55_f = np.fft.fft(imp_response['55'])
rec_50 = np.zeros((len(drs_pulses),1000))
for i in range(len(res_50)):
    out_x =  np.fft.fft(res_50[i,0])/h50_f
    out_xn = np.real(np.fft.ifft(out_x))
    rec_50[i] = scipy.signal.filtfilt(b, a, out_xn) - np.mean(scipy.signal.filtfilt(b, a, out_xn)[5:95])

for i in range(90,91):
    plt.plot(rec_50[i])
    plt.plot(res_50[i,0])
    plt.plot(res_50[i,1])

plt.plot(f0[0:500],np.abs(np.fft.fft(rec_50[i]))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,1]))[0:500])
plt.plot(f0[0:500],np.abs(np.fft.fft(res_50[i,0]))[0:500])



def recover_resonator_pulses(drs_pulses, res_frequency, ch_n, imp_response):

    res = []
    h_f = np.fft.fft(imp_response)
    
    for i in range(len(drs_pulses)):
        d1 = np.abs(np.fft.fft(drs_pulses[i,0]))
        if 40 + np.argmax(d1[40:95])==  res_frequency-1 or 40 + np.argmax(d1[40:95])==  res_frequency or 40 + np.argmax(d1[40:95])==  res_frequency+1:
            res_temp = []
            res_temp.append(drs_pulses[i,0])
            res_temp.append(drs_pulses[i,ch_n])
            out_x =  np.fft.fft(drs_pulses[i,0])/h_f
            out_xn = np.real(np.fft.ifft(out_x))
            recc = scipy.signal.filtfilt(b, a, out_xn) - np.mean(scipy.signal.filtfilt(b, a, out_xn)[5:95])
#            recc = np.concatenate( ( rec[999:], rec[0:999] ) )# to align pulses
            res_temp.append(recc)
            
            res.append(res_temp)
    return(np.asarray(res))


res_50 = recover_resonator_pulses(drs_pulses, 50-1, 1, imp_response['50'])
res_55 = recover_resonator_pulses(drs_pulses, 55-1, 2, imp_response['55'])
res_70 = recover_resonator_pulses(drs_pulses, 70, 3, imp_response['70'])


plt.plot(np.abs(np.fft.fft(res_50[456,0])))   
plt.plot(np.abs(np.fft.fft(res_50[456,1])))   
plt.plot(np.abs(np.fft.fft(res_50[456,2]))) 
  

plt.figure()
plt.plot(np.abs(np.fft.fft(res_55[4506,0])))   
plt.plot(np.abs(np.fft.fft(res_55[4506,1])))   
plt.plot(np.abs(np.fft.fft(res_55[4506,2])))  

plt.plot(res_55[4516,0])   
plt.plot(res_55[4516,1])   
plt.plot(res_55[4516,2])  


plt.plot(res_55[4504,1]) 
d = np.concatenate( ( res_55[4504,2,999:], res_55[4504,2,0:999] ) )
plt.plot(d)


plt.figure()
plt.plot(np.abs(np.fft.fft(res_70[4506,0])))   
plt.plot(np.abs(np.fft.fft(res_70[4506,1])))   
plt.plot(np.abs(np.fft.fft(res_70[4506,2]))) 



from scipy.interpolate import UnivariateSpline

import copy

def get_amp_timing(res_data):
    g_res=[]
    t_res = []
    
    for i in range(len(res_data)):#len(rec_50)
        max_arg1 = 50 + np.argmax(res_data[i,1,50:300])
        abcissa1 = np.asarray([ind for ind in range(max_arg1-4, max_arg1+3)])
        ordinate1 = res_data[i,1,abcissa1]
        spl1 = UnivariateSpline(abcissa1, ordinate1,k=3)
        xs1 = np.linspace(max_arg1-4, max_arg1+3,100)
        max_res50 = np.max(spl1(xs1))
    
        max_arg2 = 50 + np.argmax(res_data[i,2, 50:300])
        abcissa2 = np.asarray([ind for ind in range(max_arg2-4, max_arg2+3)])
        ordinate2 = res_data[i,2,abcissa2]
        spl2 = UnivariateSpline(abcissa2, ordinate2,k=3)
        xs2 = np.linspace(max_arg2-4, max_arg2+3,100)
        max_res50r = np.max(spl2(xs2))
        
        
        if np.max(spl1(xs1)) < 25000.:# and -15. < (max_res50[i,0] - max_res50[i,1])/2**16*1000 < 10.:
            g_temp =[]
            g_temp.append(max_res50)
            g_temp.append(max_res50r)
            g_res.append(g_temp)
              
            abc = np.asarray([ind for ind in range(max_arg1-4, max_arg1+1)])
            ordi = res_data[i,1,abc]
            t_temp = []
            for ii in range(len(abc)-1):
                if ordi[ii] < max_res50/2 < ordi[ii+1]:
                    yy1 = ordi[ii]
                    xx1 = abc[ii]
                    yy2 = ordi[ii+1]
                    xx2 = abc[ii+1]
                    break
                
            slope = (yy1 - yy2) / (xx1 - xx2)
            intrcept = yy1 - (slope*xx1) 
            t_temp.append(((max_res50/2) - intrcept)/slope )
        
        
            abc = np.asarray([ind for ind in range(max_arg2-4, max_arg2+1)])
            ordi = res_data[i,2,abc]
            for ii in range(len(abc)-1):
                if ordi[ii] < max_res50r/2 < ordi[ii+1]:
                    yy1 = ordi[ii]
                    xx1 = abc[ii]
                    yy2 = ordi[ii+1]
                    xx2 = abc[ii+1]
                    break
                
            slope = (yy1 - yy2) / (xx1 - xx2)
            intrcept = yy1 - (slope*xx1) 
            t_temp.append(((max_res50r/2) - intrcept)/slope )
                    
            t_res.append(t_temp)

    return np.asarray(g_res), np.asarray(t_res)
    

amp55, time55 = get_amp_timing(res_55)



#check if different fit rangesusing cubic spline for the pulse does anything? Nothing! 9050
fig7 = plt.figure()
nbins = 170
hq, bnedgess  = np.histogram(amp55[:,1],bins=np.arange(100, 22000, 100))
plt.hist(amp55[:,1], bins=np.arange(100, 22000, 100), histtype = 'step')
#yxq = 0.8*np.max(hq[110:170])*np.ones(218)
yxq1 = 0.8*484*np.ones(218)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
#plt.plot(bne11,yxq)
plt.plot(bne11,yxq1)#13603
#11560/477.3*80
hq, bnedgess  = np.histogram(amp55[:,0],bins=np.arange(100, 22000, 100))
plt.hist(amp55[:,0], bins=np.arange(100, 22000, 100), histtype = 'step')
yxq = 0.8*np.max(hq[110:170])*np.ones(218)
yxq1 = 0.8*455*np.ones(218)
bne11=(bnedgess[1:]+bnedgess[:-1])/2
plt.plot(bne11,yxq)
plt.plot(bne11,yxq1)#14300
#11820/477.3*80

plt.xlabel('charge collected \n(arb. units)',fontsize=16)
plt.ylabel('frequency',fontsize=16)
plt.tight_layout()
plt.show()


plt.figure()
plt.hist2d((amp55[:,0]-amp55[:,1])*477.3/14300, amp55[:,1]*477.3/14300, bins=[np.arange(-20, 60, 0.4), np.arange(50, 600, 1)], cmin = 1)#, bins=[np.arange(-1500, 700, 10), np.arange(30/2**12*1000, 1000/2**12*1000, 5/2**12*1000)], cmin = 1)
plt.xlabel('original peak - recovered peak \n (keV)',fontsize=16)
plt.ylabel('recovered peak \n (keV)',fontsize=16)
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.hist2d((time55[:,0]-time55[:,1])*1000, amp55[:,1]*477.3/14300, cmin = 1, bins=[np.arange(-1100, 1100, 12), np.arange(50, 600, 1)])#, bins=[np.arange(-1500, 700, 10), np.arange(30/2**12*1000, 1000/2**12*1000, 5/2**12*1000)], cmin = 1)
plt.xlabel('original timing - recovered timing \n (ps)',fontsize=16)
plt.ylabel('recovered peak \n (mV)',fontsize=16)
plt.colorbar()
plt.tight_layout()


e_range = [80., 120.,150., 200., 300., 400.,600.]
diff_amp55 = [[] for i in range(6)]
diff_t55= [[] for i in range(6)]

for i in range(len(e_range)-1):
    for j in range(len(amp55[:,1])):
        if e_range[i] < amp55[j,1]*477.3/14300 < e_range[i+1]:
            diff_amp55[i].append((amp55[j,0] - amp55[j,1])*477.3/14300)
            diff_t55[i].append((time55[j,0] - time55[j,1])*1000)

diff_ampp55 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_amp55[i])):
        if i < 6:
            if np.mean(diff_amp55[i])-50 < diff_amp55[i][j] < np.mean(diff_amp55[i])+50:
                diff_ampp55[i].append(diff_amp55[i][j])
        else:
            if np.mean(diff_amp55[i])-60 < diff_amp55[i][j] < np.mean(diff_amp55[i])+60:
                diff_ampp55[i].append(diff_amp55[i][j])            

for i in range(len(e_range)-1):
    print('mean i:', np.mean(diff_ampp55[i]))
    print('std i:', np.sqrt(np.var(diff_ampp55[i])))
#    print('mean ti:', np.mean(diff_tt50[i]))
#    print('std ti:', np.sqrt(np.var(diff_tt50[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ keV' % (np.mean(diff_ampp55[i]), ), 
        r'$\sigma=%.1f$ keV' % ( np.sqrt(np.var(diff_ampp55[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if i <6:
        fig, ax = plt.subplots()
        ax.hist(diff_ampp55[i], bins=np.arange(np.mean(diff_ampp55[i])-30, np.mean(diff_ampp55[i])+30, 1), color='gray')
        plt.xlabel('original peak - recovered peak \n (keV)')


    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    plt.ylabel('counts')
    plt.show()
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult\65MHz\charge\charge65_' + str(int(e_range[i+1]) ))

#mean i: 3.918966258028634
#std i: 5.03914113937321
#mean i: 6.0915227143958495
#std i: 4.977982998621736
#mean i: 8.6940906153429
#std i: 5.11801645326403
#mean i: 13.116968588837416
#std i: 5.4328440530152236
#mean i: 19.097205749540702
#std i: 5.772008719333687
#mean i: 24.540449191435204
#std i: 6.061975980740189



diff_tt55 = [[] for i in range(6)]
for i in range(len(e_range)-1):
    for j in range(len(diff_t55[i])):
        if i ==0 or i ==1:
            if -1100 < diff_t55[i][j] < 1000:
                diff_tt55[i].append(diff_t55[i][j])
        else:
            if np.mean(diff_t55[i])-1000 < diff_t55[i][j] < np.mean(diff_t55[i])+1000:
                diff_tt55[i].append(diff_t55[i][j])
        

for i in range(len(e_range)-1):
#    print('mean i:', np.mean(diff_amp50[i]))
#    print('std i:', np.sqrt(np.var(diff_amp50[i])))
    print('mean ti:', np.mean(diff_tt55[i]))
    print('std ti:', np.sqrt(np.var(diff_tt55[i])))
    
#    plt.hist(diff_amp50[i], bins=60, color='gray')
#    plt.xlim(np.mean(diff_amp50[i])-23, np.mean(diff_amp50[i])+23)
    textstr = '\n'.join((
        r'$\mu=%.1f$ ps' % (np.mean(diff_tt55[i]), ), 
        r'$\sigma=%.1f$ ps' % ( np.sqrt(np.var(diff_tt55[i])), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5)
    
    # place a text box in upper left in axes coords

    if 1 <= i <=3:
        fig, ax = plt.subplots()
        ax.hist(diff_tt55[i], bins=30, color='gray')
        plt.xlim(np.mean(diff_tt55[i])-600, np.mean(diff_tt55[i])+600)
        plt.xlabel('original timing - recovered timing (ps)')
        plt.show()
    elif i ==0:
        fig, ax = plt.subplots()
        ax.hist(diff_tt55[i], bins=45, color='gray')
        plt.xlim(np.mean(diff_tt55[i])-850, np.mean(diff_tt55[i])+850)
        plt.xlabel('original timing - recovered timing (ps)')
        plt.show()
    elif i ==4:
        fig, ax = plt.subplots()
        ax.hist(diff_tt55[i], bins=50, color='gray')
        plt.xlim(np.mean(diff_tt55[i])-300, np.mean(diff_tt55[i])+300)
        plt.xlabel('original timing - recovered timing (ps)')
        plt.show()
    elif i ==5:
        fig, ax = plt.subplots()
        ax.hist(diff_tt55[i], bins=50, color='gray')
        plt.xlim(np.mean(diff_tt55[i])-300, np.mean(diff_tt55[i])+300)
        plt.xlabel('original timing - recovered timing (ps)')
        plt.show()
#    elif i ==5:
#        fig, ax = plt.subplots()
#        ax.hist(diff_tt65[i], bins=35, color='gray')
#        plt.xlim(np.mean(diff_tt65[i])-700, np.mean(diff_tt65[i])+700)
#        plt.xlabel('original timing - recovered timing (ps)')
#        plt.show()
##    elif i ==1 or i==2:
##        fig, ax = plt.subplots()
##        ax.hist(diff_t50[i], bins=35, color='gray')
##        plt.xlim(np.mean(diff_t50[i])-380, np.mean(diff_t50[i])+380)
#    else:
#        fig, ax = plt.subplots()
#        ax.hist(diff_tt65[i], bins=35, color='gray')#, label = 'mean: '+str(np.mean(diff_t50[i]))
#        plt.xlim(np.mean(diff_tt65[i])-700, np.mean(diff_tt65[i])+700)
#        plt.xlabel('original timing - recovered timing (ps)')
##        plt.legend(fontsize = 12)
        
    ax.text(0.02, 0.85, textstr, fontsize=16,
             transform=ax.transAxes)#, bbox=props
    plt.ylabel('counts')
    plt.show()
    savefig(r'C:\Users\mmishra\OneDrive - North Carolina State University\drs4_multiplexerr\8to1mult\65MHz\time\time65' + str(int(e_range[i+1])) )


#mean ti: 47.63856821471259
#std ti: 207.0746595250313
#mean ti: 68.18406654467222
#std ti: 153.00665909413948
#mean ti: 82.26084053986145
#std ti: 120.18926683957514
#mean ti: 91.67587404823652
#std ti: 88.76029400856851
#mean ti: 97.97601733786878
#std ti: 66.99769600212791
#mean ti: 100.27856880800697
#std ti: 62.21592191928574


from lmfit import Model

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


gmodel = Model(gaussian)
result = gmodel.fit(y, x=x, amp=5, cen=5, wid=1)

print(result.fit_report())

plt.plot(x, y, 'bo')
plt.plot(x, result.init_fit, 'k--', label='initial fit')
plt.plot(x, result.best_fit, 'r-', label='best fit')
plt.legend(loc='best')
plt.show()