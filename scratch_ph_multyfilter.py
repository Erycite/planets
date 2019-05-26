# from scipy import signal as signal
# from scipy.signal import find_peaks
# import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib.colors as colors
# from MorletWavelet_1605 import MorletWavelet
#
# ########################## Take signal and sr ##############################
# def f_peaks(Y, sr, smoothing_factor, thresh):
#     # dt = 1/sr
#     # for j in range(1, len(Y)):
#     #     dy.append((Y[j] - Y[j - 1]) / dt)
#     dy = []
#     dy = np.diff(Y)*sr
#     #ys = smooth(np.asarray(dy), window_len=smoothing_factor, window='flat', corrected=True)
#     ys = smooth(np.asarray(dy), smoothing_factor)
#     pk = []
#     for i in range(len((ys[:-1]))):
#         if ys[i] > 0 and ys[i + 1] < 0 and Y[i]> thresh:
#             pk.append(i)
#     return(pk)
#
# def _analog_filter(data, sr, ftype, highpass=None, lowpass=None, order=3, zerophase=True):
#     from scipy import signal
#     # Project: bark   Author: kylerbrown   File: stream.py
#     ' Use a classic analog filter on the data, currently butter or bessel'
#     from scipy.signal import butter, bessel
#     filter_types = {'butter': butter, 'bessel': bessel}
#     afilter = filter_types[ftype]
#     if highpass is None and lowpass is not None:
#         b, a = afilter(order, lowpass / (sr / 2), btype='lowpass')
#     elif highpass is not None and lowpass is None:
#         b, a = afilter(order, highpass / (sr / 2), btype='highpass')
#     elif highpass is not None and lowpass is not None:
#         if highpass < lowpass:
#             b, a = afilter(order, (highpass / (sr / 2), lowpass / (sr / 2)), btype='bandpass')
#         else:
#             b, a = afilter(order, (lowpass / (sr / 2), highpass / (sr / 2)), btype='bandstop')
#     if zerophase:
#         y = signal.filtfilt(b, a, data)
#         return y
#     else:
#         y = signal.lfilter(b, a, data)
#         return y
#
# def smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
#
# def rwrkd_ipi(pks, s_fact):
#     ipksi = np.diff(pks)
#     #sipi = smooth_1(ipksi, window_len=s_fact, window='flat', corrected=True)
#     sipi = smooth(ipksi, 11)
#     #u = [np.repeat(dpk, dpk) for dpk in sipi]
#     u = [np.repeat(dpk, ipksi[i]) for i, dpk in enumerate(sipi)]
#     u = np.concatenate(u)
#     rwu = np.concatenate(
#         [np.repeat(np.mean(u), pks[0]), u[:], np.repeat(np.mean(u), (len(s1) - pks[-1]))])
#     return(rwu)
#
# def find_Candidate_Epoch(rwu, p10):
#     per10 = np.percentile(rwu, p10)
#
#     ceps = np.where(rwu <= per10)[0]
#     SCep = [ceps[0]]
#     FCep = []
#     Cep = []
#     for x in range(1, len(ceps)):
#         if (ceps[x] - ceps[x - 1]) != 1:
#             SCep.append(ceps[x])
#
#     for x in range(len(SCep)):
#         if any(rwu[SCep[x]:] > per10):
#             FCep.append(SCep[x] + np.where(rwu[SCep[x]:] > per10)[0][0])
#             Cep.append([SCep[x], FCep[x]])
#         else:
#             print('never rise again')
#     return(Cep)
#
# def find_phasic(Cep, st_power, p5):
#     per5 = np.percentile(rwu, p5)
#     phasic_ep = []
#     nph_ep = []
#     for cep in Cep:
#         if cep[1] - cep[0] < min_ph_duration:  # 900ms in paper (rat)
#             print('epoch is too short' + str(cep))
#             nph_ep.append([cep, 'tooshort'])
#         elif rwu[cep[0]:cep[1]].min() > per5:
#             print('nerver < to 5percentile inter-peak-interval')
#             nph_ep.append([cep, 'toohigh_ipi'])
#         elif np.mean(st_power[cep[0]:cep[1]]) < np.mean(st_power):
#             print('power is too low (< mean power in the full REM episod)')
#             nph_ep.append([cep, 'toolow_thetapower'])
#         else:
#             print('Candidate epoch is validated !! ')
#             print(cep)
#             phasic_ep.append(cep)
#     #return(phasic_ep, nph_ep)
#     return(phasic_ep)
#
# def ph_vect(phasic_epo, len_sig):
#     ph_v = np.repeat(0, len_sig)
#     for ep in phasic_epo:
#         ph_v[ep[0]:ep[1]] =1
#     return(ph_v)
#
# ypos = [8,6,8,10]
# ypos = [4,10,np.mean(bands),np.mean(bands)]
# colos = ['b', 'r', 'm', 'orange']
# ls = ['--', '-.', ':','-']
#
# sr = 1000
# #datapath = '/media/matias/3f83b50b-cd10-45f4-8814-86b79cbad59e/lfsc_reorganized/'
# datapath = '/home/mathias/Documents/LFSC/'
#
# S = np.load((datapath + 'C4_SP.npz'))['dCA1']
#
# #########################################################
# # Params to play with
# p10 = 10 # Percentile (equiv to percentile10 in the paper), Here we take 15 to be more permissive
# p5 = 5 # Percentile 5 equivalent
#
# #bands = [[4,12], [4,8], [6,10], [8,12]]
# #bands = [[5,12], [6,11], [7,10]]# Up to 4 different bands yet
# bands = [[4,8], [7,11]]
#
#
# min_ph_duration = 900 # 900ms according to the paper, but btwn [500-900] the results are interesting
# f_peak_sensi =95 #percentile for pks,_=find_peaks(st_power, height=np.percentile(st_power, f_peak_sensi), distance= 11)
# # f_peak_sensi = 95 seems the best ! [92-98] seems ok/Good!
# #f_peak_dist = 25 # same as before, Almost no impact, so keep it relatively hight [10-100] to avoid errors
# wl = 11 # Smoothing factor
# #### Morlet wavelet params ####
# freq_band = [4,18]
# n_freq = 28 # Useless to play with
# range_cycle = [10,30]#[16,40] #Previously [4,40], works well with [8-32], but we choose even higher
#
#
#
#
# S = np.load((datapath + 'C5_SP.npz'))['dCA1']
# s1 = S[3]
#
# for s1 in S[2:3]:
#
#     B = [ [[5,7.5], [7,5,10], [5,10]], [[4,7], [7,12], [4,12]], [[4,7.5], [7.5,11], [4,11]]]
#     for bands in B:
#         #bands = [[4,8], [7,11]]
#         plt.figure()
#         ##############################################################################################
#         ##############################################################################################
#         tf, time, frex = MorletWavelet(s1, freq_band, n_freq, range_cycle)
#         x, y = np.meshgrid(time, frex)
#         ###########################################
#         for i, band in enumerate(bands):
#             binf = band[0]
#             bsup = band[1]
#
#             St = _analog_filter(s1, ftype='butter', sr=1000, highpass=binf, lowpass=bsup, order=3, zerophase=True)
#             sh = signal.hilbert(St)
#             st_power = np.abs(sh)**2
#             pks = np.asarray(f_peaks(st_power, sr, 40, np.percentile(st_power, 75)))
#             stp = (st_power / st_power.max()) * 2 + ypos[i] ### Normalized to be plotted on the scaleogram
#             ## return stp, pks
#
#
#             rwu = rwrkd_ipi(pks, 11)
#             # return smoothed inter peak interval
#
#             Candidate_epoch = find_Candidate_Epoch(rwu, p10)
#             phasic_epo = find_phasic(Candidate_epoch, st_power, p5)
#             # return accepted epochs
#             ph_v = ph_vect(phasic_epo, len(s1))
#             # return band
#
#
#             plt.subplot(311)
#             plt.title(str(bands))
#             plt.contourf(x, y, tf, norm=colors.PowerNorm(gamma=1. / 2., vmin=tf.min(), vmax=tf.max()))
#             plt.semilogy()
#             plt.plot(time, stp, color=colos[i], linewidth=0.8)
#             plt.plot(time[pks], stp[pks], '.', color= colos[i])
#             plt.hlines(band, 0, time.max(), color=colos[i], linestyles= ls[i])
#             for ph in phasic_epo:
#                 plt.hlines( (12+ (ypos[i]/10)), ph[0]*(1/sr), ph[1]*(1/sr), color=colos[i])
#
#
#
#             plt.subplot(312)
#             plt.plot(rwu, color=colos[i], lw=0.7)
#             #plt.plot(ipi*10000+4)
#             plt.hlines(np.percentile(rwu, 10), 0, len(rwu), color=colos[i])
#             plt.vlines(phasic_epo[:][:], 0, rwu.max(), color=colos[i], linestyles= '--')
#
#
#
#             perc_ph = round(np.sum(ph_v) / len(ph_v) * 100 , 2)
#             txt = (str(perc_ph) + '% of Phasic in this ep')
#
#             tf_ph = np.mean(tf[:, np.where(ph_v != 0)[0]], axis=1)
#             tf_to = np.mean(tf[:, np.where(ph_v == 0)[0]], axis=1)
#
#             plt.subplot(3,len(bands),(2*len(bands)+1)+i)
#             plt.plot(frex, tf_to)
#             plt.plot(frex, tf_ph, colos[i])
#             plt.legend(['Tonic', 'Phasic'])
#             plt.title(txt, y=-0.5)
#





from scipy import signal as signal
from scipy.signal import find_peaks
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from MorletWavelet_1605 import MorletWavelet

def list_of_suj(take_suj):
    """
    :param take_suj: can be str 'all_suj' or 'CEFG' (select the group desired), or a list of the desired subjects
    Use the names from the folders name in the main data folder
    :return: Lsuj: Return a list of the subjects to work with
    """

    ListDir = os.listdir('/home/mathias/Documents/LFSC/')
    allsuj = sorted([x for x in ListDir if len(x) < 3])

    if type(take_suj) is list:
        Lsuj = take_suj
        #print('we took those suj', Lsuj)
    if type(take_suj) is str:

        if take_suj == 'all_suj':
            Lsuj = allsuj
            #print('we took all suj')
        else:
            Lsuj = []
            for grp in take_suj:
                #print('letter looked is ', grp)
                if grp == 'C':
                    Lsuj += allsuj[0:8]
                    #print('we add the group C')
                elif grp == 'E':
                    Lsuj += allsuj[8:15]
                    #print('we add the group E')
                elif grp == 'F':
                    Lsuj += allsuj[15:17]
                    #print('we add the group F')
                elif grp == 'G':
                    Lsuj += allsuj[17:19]
                    #print('we add the group G')
    Lsuj = sorted(Lsuj)
    return(Lsuj)



######################### Take signal and sr ##############################
def f_peaks(Y, sr, smoothing_factor, thresh):
    # dt = 1/sr
    # for j in range(1, len(Y)):
    #     dy.append((Y[j] - Y[j - 1]) / dt)
    dy = []
    dy = np.diff(Y)*sr
    #ys = smooth(np.asarray(dy), window_len=smoothing_factor, window='flat', corrected=True)
    ys = smooth(np.asarray(dy), smoothing_factor)
    pk = []
    for i in range(len((ys[:-1]))):
        if ys[i] > 0 and ys[i + 1] < 0 and Y[i]> thresh:
            pk.append(i)
    return(pk)

def f_pks(Y, thresh):
    ys = np.diff(Y)
    pk = []
    for i in range(len((ys[:-1]))):
        if ys[i] > 0 and ys[i + 1] < 0 and Y[i]> thresh:
            pk.append(i+1)
    return(pk)

def _analog_filter(data, sr, ftype, highpass=None, lowpass=None, order=3, zerophase=True):
    from scipy import signal
    # Project: bark   Author: kylerbrown   File: stream.py
    ' Use a classic analog filter on the data, currently butter or bessel'
    from scipy.signal import butter, bessel
    filter_types = {'butter': butter, 'bessel': bessel}
    afilter = filter_types[ftype]
    if highpass is None and lowpass is not None:
        b, a = afilter(order, lowpass / (sr / 2), btype='lowpass')
    elif highpass is not None and lowpass is None:
        b, a = afilter(order, highpass / (sr / 2), btype='highpass')
    elif highpass is not None and lowpass is not None:
        if highpass < lowpass:
            b, a = afilter(order, (highpass / (sr / 2), lowpass / (sr / 2)), btype='bandpass')
        else:
            b, a = afilter(order, (lowpass / (sr / 2), highpass / (sr / 2)), btype='bandstop')
    if zerophase:
        y = signal.filtfilt(b, a, data)
        return y
    else:
        y = signal.lfilter(b, a, data)
        return y

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def rwrkd_ipi(pks, s_fact):
    if s_fact >= len(pks):
        s_fact = 2
    ipksi = np.diff(pks)
    #sipi = smooth_1(ipksi, window_len=s_fact, window='flat', corrected=True)
    sipi = smooth(ipksi, s_fact)
    #u = [np.repeat(dpk, dpk) for dpk in sipi]
    u = [np.repeat(dpk, ipksi[i]) for i, dpk in enumerate(sipi)]
    u = np.concatenate(u)
    rwu = np.concatenate(
        [np.repeat(np.mean(u), pks[0]), u[:], np.repeat(np.mean(u), (len(s1) - pks[-1]))])
    return(rwu)

def find_Candidate_Epoch(rwu, p10):
    per10 = np.percentile(rwu, p10)

    ceps = np.where(rwu <= per10)[0]
    SCep = [ceps[0]]
    FCep = []
    Cep = []
    for x in range(1, len(ceps)):
        if (ceps[x] - ceps[x - 1]) != 1:
            SCep.append(ceps[x])

    for x in range(len(SCep)):
        if any(rwu[SCep[x]:] > per10):
            FCep.append(SCep[x] + np.where(rwu[SCep[x]:] > per10)[0][0])
            Cep.append([SCep[x], FCep[x]])
        else:
            print('never rise again')
    return(Cep)

def find_phasic(Cep, st_power, p5):
    per5 = np.percentile(rwu, p5)
    phasic_ep = []
    nph_ep = []
    for cep in Cep:
        if cep[1] - cep[0] < min_ph_duration:  # 900ms in paper (rat)
            print('epoch is too short' + str(cep))
            nph_ep.append([cep, 'tooshort'])
        elif rwu[cep[0]:cep[1]].min() > per5:
            print('nerver < to 5percentile inter-peak-interval')
            nph_ep.append([cep, 'toohigh_ipi'])
        elif np.mean(st_power[cep[0]:cep[1]]) < np.mean(st_power):
            print('power is too low (< mean power in the full REM episod)')
            nph_ep.append([cep, 'toolow_thetapower'])
        else:
            print('Candidate epoch is validated !! ')
            print(cep)
            phasic_ep.append(cep)
    #return(phasic_ep, nph_ep)
    return(phasic_ep)

def ph_vect(phasic_epo, len_sig):
    ph_v = np.repeat(0, len_sig)
    for ep in phasic_epo:
        ph_v[ep[0]:ep[1]] =1
    return(ph_v)


ypos = [15,16,17,11.5,12,12.5,4.5,5,5.5]
colos = ['#a9425f','#9fb843','#49c2a0','#6b51ca', '#326881','#7e3490','#9f5635','#4cae35','#42491a' ]
ls = ['-', '--', ':', '-.']

sr = 1000
#datapath = '/media/matias/3f83b50b-cd10-45f4-8814-86b79cbad59e/lfsc_reorganized/'
datapath = '/home/mathias/Documents/LFSC/'

S = np.load((datapath + 'C6_SP.npz'))['dCA1']

#########################################################
# Params to play with
p10 = 10 # Percentile (equiv to percentile10 in the paper), Here we take 15 to be more permissive
p5 = 5 # Percentile 5 equivalent

min_ph_duration = 900 # 900ms according to the paper, but btwn [500-900] the results are interesting
f_peak_sensi =95 #percentile for pks,_=find_peaks(st_power, height=np.percentile(st_power, f_peak_sensi), distance= 11)
# f_peak_sensi = 95 seems the best ! [92-98] seems ok/Good!
#f_peak_dist = 25 # same as before, Almost no impact, so keep it relatively hight [10-100] to avoid errors
wl = 11 # Smoothing factor
#### Morlet wavelet params ####
freq_band = [4,18]
n_freq = 28 # Useless to play with
range_cycle = [10,30]#[16,40] #Previously [4,40], works well with [8-32], but we choose even higher
#Bands
a = [4,5,6]
b = [7,8,9]
c = [10,11,12]
ab = [[x,y] for x in a for y in b]
ac = [[x,y] for x in a for y in c]
bc = [[x,y] for x in b for y in c]
Lsuj=['C2']#,'C3','C4']
for suj in Lsuj:
    S = np.load((datapath + str(suj) +'_SP.npz'))['dCA1']

    B = [ab,ac, bc]
    suj_phvs =[]
    for s1 in S[:]:
        phv = []
        for bands in B:
            #bands = [[4,8], [7,11]]
            # plt.figure()
            ##############################################################################################
            ##############################################################################################
            tf, time, frex = MorletWavelet(s1, freq_band, n_freq, range_cycle)
            x, y = np.meshgrid(time, frex)

            ###########################################
            for i, band in enumerate(bands):
                # iselect = [0,1,2,3]*20
                # i = iselect[band_ind]
                j = np.repeat([0,1,2,3],4)[i]

                binf = band[0]
                bsup = band[1]

                St = _analog_filter(s1, ftype='butter', sr=1000, highpass=binf, lowpass=bsup, order=3, zerophase=True)
                sh = signal.hilbert(St)
                st_power = np.abs(sh)**2
                pks = np.asarray(f_peaks(st_power, sr, 40, np.percentile(st_power, 75)))
                stp = (st_power / st_power.max()) * 2 + ypos[i] ### Normalized to be plotted on the scaleogram
                ## return stp, pks


                rwu = rwrkd_ipi(pks, 11)
                # return smoothed inter peak interval

                Candidate_epoch = find_Candidate_Epoch(rwu, p10)
                phasic_epo = find_phasic(Candidate_epoch, st_power, p5)
                # return accepted epochs
                ph_v = ph_vect(phasic_epo, len(s1))
                # return band

                phv.append(ph_v)
        #suj_phvs.append(phv)

                # plt.subplot(411)
                # plt.title(str(bands))
                # plt.contourf(x, y, tf, norm=colors.PowerNorm(gamma=1. / 2., vmin=tf.min(), vmax=tf.max()))
                # plt.semilogy()
                # # plt.plot(time, stp, color=colos[i], linewidth=0.8)
                # # plt.plot(time[pks], stp[pks], '.', color= colos[i])
                # plt.hlines(band, 0, time.max(), color=colos[i], linestyles= ls[j], lw=0.5)
                # for ph in phasic_epo:
                #     plt.hlines( ((ypos[i])), ph[0]*(1/sr), ph[1]*(1/sr), color=colos[i], linestyle=ls[j], lw=3)

                perc_ph = round(np.sum(ph_v) / len(ph_v) * 100 , 2)
                txt = (str(perc_ph) + '% -'+str(band))

                tf_ph = np.mean(tf[:, np.where(ph_v != 0)[0]], axis=1)
                tf_to = np.mean(tf[:, np.where(ph_v == 0)[0]], axis=1)

                # plt.subplot(3,4,(4+1)+i)
                # plt.subplot(4, 3, (3 + 1) + i)
                # plt.plot(frex, tf_to)
                # plt.plot(frex, tf_ph, colos[i], linestyle=ls[j])
                # plt.legend(['Tonic', 'Phasic'])
                # plt.title(txt, y=0.2, x=0.75)
        suj_phvs.append(phv)
    #[plt.plot(np.transpose(phv[x])) for x in range(len(phv))]
    rfact = 2
    plt.figure()
    all_ep_phv = np.concatenate(suj_phvs, axis=1)
    all_ep_phv = all_ep_phv[::2]
    all_ep_phv  = np.ma.masked_where(all_ep_phv  < 1 , all_ep_phv )
    [plt.plot(np.transpose(all_ep_phv[x])*(1+(x/10))) for x in range(len(all_ep_phv))]
    A = [len(suj_phvs[x][0])for x in range(len(suj_phvs))]
    plt.vlines([sum(A[:x]) for x in range(len(A))], 1, (1+len(all_ep_phv)/10), linestyle='--', lw=2, zorder=0)
    BB = [str(Bx) for Bx in np.concatenate(B)]
    plt.yticks(np.arange(1,(1+len(all_ep_phv)/10),step=0.1 ),BB)


a_phv = np.concatenate(suj_phvs, axis=1)
L_allph = len(a_phv[0])
perc = [str(round(100*sum(a_phv[x])/L_allph,2) + '%') for x in range(len(a_phv))]
plt.legend(perc[::-1]) # make sure list is in good order !



# ######## Looking for the most common frequency peak in the data
# fpk = []
# Lsuj = list_of_suj('C')
# for suj in Lsuj:
#     S = np.load((datapath + suj+'_SP.npz'))['dCA1']
#
#     for s1 in S:
#         f, Px = signal.welch(s1, fs=sr, window='hanning', nperseg=4*sr, noverlap=2*sr, scaling='spectrum')
#         #plt.figure()
#         #plt.semilogx(f, Px)
#         #plt.vlines([4,12], 0, Px.max())
#         #psp = f_pks(Px, np.percentile(Px[10:70],85))
#         psp = find_peaks(Px[10:70], height= np.percentile(Px[10:70], 95))
#         psp = psp[0]+10
#         #plt.hlines( np.percentile(Px[10:70], 85), f[10], f[70])
#         #plt.semilogx(f[psp], Px[psp], '.')
#
#         fpk.append(f[psp])
#
#
#     fpk= [np.concatenate(fpk)]
#
# fpk = np.concatenate(fpk)
#
# plt.figure()
# plt.hist(fpk, bins=len(np.unique(fpk)))