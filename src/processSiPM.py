#!/usr/bin/python3

import numpy as np
import h5py
import sys
import re

def main():
	stride = 4100
	if len(sys.argv)<2:
		print('syntax: sys.argv[0] <h5 filelist, space separated>')
		return
	hbins = [i for i in range(-2048,2049)]
	h = np.zeros(len(hbins)-1,dtype=int)
	rbins = [4*i for i in range(-2048,2049)]
	r = np.zeros(len(rbins)-1,dtype=int)
	for fname in sys.argv[1:]:

		m = re.search('(\w+)/(\w+)(\d{5})\.h5',fname)
		if m:
			path = m.group(1)
			filefront = m.group(2)
			filenum = int(m.group(3))
			with h5py.File(fname,'r') as f:
				print(list(f['Waveforms']['Channel 4'].keys()))
				data = f['Waveforms']['Channel 4']['Channel 4Data'][()]
				r += np.histogram(data,rbins)[0]
				cols = data.shape[0]//stride
				data = data[:cols*stride].reshape(cols,stride).T
				fmat = np.tile(np.fft.fftfreq(cols),(stride,1))
				DATA = np.fft.fft(data,axis=1)*1j*fmat
				back = np.fft.ifft(DATA).real.astype(int)
				outname = '%s/processed/%s_%06i.out'%(path,filefront,filenum)
				np.savetxt(outname,data,fmt='%i')
				h += np.histogram(back.reshape(-1),hbins)[0]

	outname = '%s/processed/%s.rawhist'%(path,filefront)
	np.savetxt(outname,r,fmt='%i')
	outname = '%s/processed/%s.hist'%(path,filefront)
	np.savetxt(outname,h,fmt='%i')
	return

if __name__ == '__main__':
    main()
