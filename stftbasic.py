import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fftpack import fft
import csv
ofile=open('basicCxmX.csv', "wb")
writer=csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

import sys,math,os
sys.path.append('/Users/mac/git/sms-tools/software/models')
import utilFunctions as UF
#import stft as STFT
import dftModel as DFT

w = 'hamming'
M = 2205
N = 4096
H = 2205
maxFreq = 1000.0

# M = 131072
# N = 131072
# H = 44100*4

(fs, x) = UF.wavread('/Users/mac/git/sms-tools/sounds/basicCmaj.wav')
#(fs, x) = scipy.io.wavfile.read('/Users/mac/git/sms-tools/sounds/basicCmaj.wav', mmap=False)

w = get_window(w, M)

# m, p = STFT.stftAnal(x, w, N, H)


M = w.size                                      
hM1 = int(math.floor((M+1)/2))                  
hM2 = int(math.floor(M/2))                      
#x = np.append(np.zeros(hM2),x)                  
#x = np.append(x,np.zeros(hM2))                  
pin = hM1                                            
pend = x.size-hM1                               
w = w / sum(w)                                  
while pin<=pend:                                      
	x1 = x[pin-hM1:pin+hM2]                       
	mX, pX = DFT.dftAnal(x1, w, N)                
	if pin == hM1:                               
		xmX = np.array([mX])
		xpX = np.array([pX])
		#writer.writerow([xmX])
	else:                                        
		xmX = np.vstack((xmX,np.array([mX])))
		xpX = np.vstack((xpX,np.array([pX])))
		#writer.writerow([xmX])
	pin += H                                     
	# return mX, pX

print "Printing DFT:"
#print type(mX)
print np.array([mX])
np.savetxt("xmXtext.txt",xmX,delimiter=' ')

#print "Printing stacked frames:"
#print xmX
#plt.plot(np.arange(x.size)/float(fs), x)
#plt.plot(mX)
#plt.pcolormesh(np.transpose(xmX))

numFrames = int(xmX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = fs*np.arange(N*maxFreq/fs)/N
plt.pcolormesh(frmTime, binFreq, np.transpose(xmX[:,:N*maxFreq/fs+1]))
plt.show()