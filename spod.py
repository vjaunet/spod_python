#
#          SPOD calculation
#
# ==============================================================

import numpy as np
from scipy.signal import csd,hann

def SPOD (q,nfft=256,Stsampling=1.0):
    """ q[nt,nx] : 2D array containing space and time information """

    # Prepare some temporary tables >>>>>>>>>>>>>>>>>>>>>>>>>>>
    np.size(q,1)
    qz=np.tile(q,np.size(q,1))          #repeat the whole table nx times
    qx=np.repeat(q,np.size(q,1),axis=1) #repeat each xposition nx times

    # compute the CSD >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    [St,CSD]=csd(qx, qz, fs=Stsampling,
                 window='hann', nperseg=nfft, noverlap=nfft/2, nfft=nfft,
                 detrend=False, return_onesided=False,  axis=0)
    CSD=np.reshape(CSD,[CSD.shape[0],q.shape[1],q.shape[1]])

    St=np.fft.fftshift(St)
    CSD = np.fft.fftshift(CSD,axes=0)

    # Compute POD >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    sigma,SPODmode = np.linalg.eig(CSD)

    # sort the eigenvalues and re-order the vectors >>>>>>>>>>>>
    sigma,SPODmode = sortSPOD(sigma,SPODmode)

    return St, sigma, SPODmode

def SPOD_LW (q,nfft=256,Stsampling=1.0):
    """
    Computes the SPOD of a table using a lightweight algorithm
    of Towne et al. (2017)

    q[nt,nx] : 2D array containing space and time information

    """
    nx=q.shape[1]
    nblocs=int(np.floor(q.shape[0]/nfft))*2 #number of nfft blocs + overlap

    # Compute the FFT realizations >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ibloc=0; ideb=0; ifin=nfft
    qhat=np.zeros([nfft,nx,nblocs],dtype='complex64')
    w = np.tile(np.hanning(nfft),(nx,1)).transpose()
    while ifin < q.shape[0]:
        qhat[:,:,ibloc] = np.fft.fft(w*q[ideb:ifin,:],axis=0)
        qhat[:,:,ibloc] = np.fft.fftshift(qhat[:,:,ibloc],axes=0)
        ideb+=int(nfft/2)
        ifin+=int(nfft/2)
        ibloc+=1

    #Computing density and normalizing by the window energy
    qhat = qhat/ibloc * Stsampling/nfft /np.hanning(nfft).std()**2

    Momega=np.zeros([nfft,nblocs,nblocs],dtype='complex64')
    for iomega in np.arange(nfft):
        Momega[iomega,:,:] = qhat[iomega,:,:].T.conj() @ qhat[iomega,:,:]

    # compute the eigen value problem
    sigma,eigVec = np.linalg.eig(Momega)

    # sort the eigenvalues and re-order the vectors >>>>>>>>>>>>
    sigma,eigVec = sortSPOD(sigma,eigVec)

    # build the SPOD modes
    SPODmode=np.zeros([nfft,nx,nblocs],dtype='complex64')
    for iomega in np.arange(nfft):
       SPODmode[iomega,:,:] = (qhat[iomega,:,:] @
                             eigVec[iomega,:] @
                             np.diag(np.real(sigma[iomega,:]+1e-10)**(-1/2)))

    St = np.linspace(-Stsampling/2,Stsampling/2,nfft)
    return St, sigma, SPODmode

def sortSPOD(sig,eigVec):
    """
     Sorting Eigenvalues by ascending order
         and Eigenmodes respectively
    """
    i = np.arange(len(sig))[:, np.newaxis] #list of lines
    j = sig.argsort()[:,::-1]  #sorted list of columns in decreasing order
    sig=sig[i,j] #sorting sigma according to the indexing

    ii = np.arange(eigVec.shape[1])[:, np.newaxis] #list of lines
    for ifreq in i:
        jj=j[ifreq,:].repeat(eigVec.shape[1],axis=0)
        eigVec[ifreq,:,:] = eigVec[ifreq,ii,jj]

    return sig,eigVec
