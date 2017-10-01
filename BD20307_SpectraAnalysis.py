#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:38:35 2017

@author: maggiethompson
"""

from __future__ import unicode_literals
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import fits
from scipy.stats import chisquare
import pandas as pd 
from scipy import interpolate

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11.5}

matplotlib.rc('font', **font)


fig, ax = plt.subplots()

#Upload SOFIA normalized, photosphere subtracted spectrum

sofiaspec = np.loadtxt('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIA_photsub_FINAL.txt', skiprows=1)
sofia_wave = sofiaspec[:,0]
sofia_flux=(sofiaspec[:,1])

sofia_err = sofiaspec[:,2]
sofia_fluxunc=(sofia_err/sofia_flux)**2
sofia_percunc=np.sqrt((sofia_fluxunc)+(0.0522/1.298)**2)
sofia_errtot=sofia_percunc * sofia_flux 

ax.errorbar(sofia_wave, sofia_flux, yerr=sofia_errtot, linewidth=1.0)

plt.title(r'BD +20 307 Average Spectra')
plt.xlabel(r'wavelength ($\mu$m)')
plt.ylabel(r'weighted average flux (Jy)')


blue_patch = mpatches.Patch(color='blue', label=r'SOFIA Spectrum')
red_patch = mpatches.Patch(color='red', label=r'Ground-based Spectrum')
green_patch = mpatches.Patch(color='green', label=r'Spitzer Spectrum')

#Upload ground-based (Keck & Gemini) normalized and photosphere subtracted spectrum
#ORIGINAL GROUND SPECTRUM (NOT PHOTSUB/NORM)
grounddata_original=fits.open('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/lwsandmichellespec.fits')
tbdata_original=grounddata_original[0].data
groundwave_original=tbdata_original[1]
groundflux_original=tbdata_original[0]
grounderr_original=tbdata_original[2]
#PHOTOSPHERE SUBTRACTED/NORMALIZED GROUND SPECTRUM
groundtxt=np.loadtxt('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/lwsmichelle_photsub.tab', skiprows=1)
ground_data=np.array(groundtxt)
groundwave=ground_data[:,0]
groundflux=ground_data[:,1]
grounderr=ground_data[:,2]
ax.errorbar(groundwave, groundflux, yerr=grounderr, color='r', label='Ground-based Spectrum', linewidth=1.0)

#Upload the normalized, photosphere subtracted Spitzer spectrum

#Spitzer Spectrum
spitzertxt = np.loadtxt('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/bd20307_mergespec_norm_mips1_photsub.tab', skiprows=2)
spitzerwave = spitzertxt[:,0]
spitzerflux = spitzertxt[:,1]
spitzererr = spitzertxt[:,2]
ax.errorbar(spitzerwave, spitzerflux, yerr=spitzererr, color='g', linewidth=1.0)
#ax.plot(spitzerwave, spitzerflux, color='g')
ax.legend(handles=[blue_patch, red_patch, green_patch], loc='upper right', fontsize='small')
ax.set_xlim([8,14])
#plt.savefig('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIA_Spitz_Ground.png', dpi=1000)
plt.show()

#Plotting the original ground-based spectrum and the normalized, photosphere-subtracted spectrum

fig,ax=plt.subplots()
ax.errorbar(groundwave_original, groundflux_original, yerr=grounderr_original, color='g', linewidth=1.0)
ax.errorbar(groundwave, groundflux, yerr=grounderr, color='b', linewidth=0.5)
greenlabel = mpatches.Patch(color='green', label=r'Original')
bluelabel = mpatches.Patch(color='blue', label=r'Normalized & Phot-Sub')
ax.legend(handles=[greenlabel, bluelabel], loc='upper right', fontsize='small')
plt.title('Comparison of Keck-Gemini Spectra')
#plt.ylim(1, 1.5)
#lt.xlim(9.5, 12)
plt.show()
#--------------------------ANALYSIS-------------------------------------------------------

#Now we want to compare the three spectra to see if we can conclude how BD +20 307's spectrum has changed over time

#chi_SOFspitz=np.sum(((sofia_flux - 1.11697)**2)/(sofia_errtot**2))

#chi_groundspitz=np.sum(((groundflux - 1.11697)**2)/(grounderr**2))

subsetSOF_flux=[]
subsetSOF_wave=[]
subsetSOF_unc=[]
for (x,y,z) in zip(sofia_wave,sofia_flux,sofia_errtot):
    if (x >= 9) & (x <= 12):
        #print x,y   
        subsetSOF_flux.append(y)
        subsetSOF_wave.append(x)
        subsetSOF_unc.append(z)
subsetSOF_wave_n=np.around(subsetSOF_wave, decimals=1)

subsetG_flux=[]
subsetG_wave=[]
subsetG_unc=[]
for (x,y,z) in zip(groundwave,groundflux,grounderr):
    if (x >= 9) & (x <= 12):
        #print x,y
        subsetG_flux.append(y)
        subsetG_wave.append(x)
        subsetG_unc.append(z)
subsetG_wave_n=np.around(subsetG_wave, decimals=1)
        
subsetSpitz_flux=[]
subsetSpitz_wave=[]
subsetSpitz_unc=[]
for (x,y,z) in zip(spitzerwave,spitzerflux,spitzererr):
    if (x >= 9) & (x <= 12):
        #print x,y 
        subsetSpitz_flux.append(y)
        subsetSpitz_wave.append(x)
        subsetSpitz_unc.append(z)
subsetSpitz_wave_n=np.around(subsetSpitz_wave, decimals=1)

#Creating dataframes for 9-12 micron spectral data
SOF_data={'wave':subsetSOF_wave, 'flux':subsetSOF_flux, 'unc':subsetSOF_unc}
SOF_df = pd.DataFrame(data=SOF_data)

G_data={'wave':subsetG_wave, 'flux':subsetG_flux, 'unc':subsetG_unc}
G_df=pd.DataFrame(data=G_data)

Spitz_data={'wave':subsetSpitz_wave, 'flux':subsetSpitz_flux, 'unc':subsetSpitz_unc}
Spitz_df=pd.DataFrame(data=Spitz_data)

#merged=pd.merge(SOF_df, Spitz_df, on=['wave', 'wave'], how='right')

#SOF_df.to_csv('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIA.csv')
#G_df.to_csv('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/ground.csv')
#Spitz_df.to_csv('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/spitz.csv')

#combo_data=pd.read_csv('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOF_Spitzer_Differences.csv', na_values='NaN')

#Interpolating Spitzer and ground-based spectra to SOFIA spectrum (only doing this over the subset of spectral data from 9-12 microns)

f=interpolate.interp1d(Spitz_df.wave, Spitz_df.flux, bounds_error=False)
#f2=interpolate.interp1d(Spitz_df.wave, Spitz_df.flux, kind='cubic')

Spitz_wavenew=SOF_df.wave
Spitz_fluxnew=f(Spitz_wavenew)

f_unc=interpolate.interp1d(Spitz_df.wave, Spitz_df.unc, bounds_error=False)
Spitz_uncnew=f_unc(Spitz_wavenew)

g=interpolate.interp1d(G_df.wave, G_df.flux, bounds_error=False)
G_wavenew=SOF_df.wave
G_fluxnew=g(G_wavenew)

g_unc=interpolate.interp1d(G_df.wave, G_df.unc, bounds_error=False)
G_uncnew=g_unc(G_wavenew)

fig, ax = plt.subplots()
ax.errorbar(SOF_df.wave, SOF_df.flux, SOF_df.unc, linewidth=1.0, color='b')
ax.errorbar(G_wavenew, G_fluxnew, G_uncnew, linewidth=1.0, color='r')
ax.errorbar(Spitz_wavenew, Spitz_fluxnew, Spitz_uncnew, linewidth=1.0, color='g')
plt.xlabel('wavelength (microns)')
plt.ylabel('average flux (Jy)')
plt.title('BD +20 307 Spectra (Spitzer and ground-based spectra linearly interpolated over 9-12 microns)')
blue_patch = mpatches.Patch(color='blue', label='$SOFIA$ $Spectrum$')
red_patch = mpatches.Patch(color='red', label='$Ground-based$ $Spectrum$')
green_patch = mpatches.Patch(color='green', label='$Spitzer$ $Spectrum$')
ax.legend(handles=[blue_patch, red_patch, green_patch], loc='upper right', fontsize='small')
plt.show()

SOFSpitz_diff=(SOF_df.flux/Spitz_fluxnew)
SOFSpitz_diffavg=np.mean(SOFSpitz_diff)
SOFSpitz_diffunc = SOFSpitz_diff * np.sqrt((SOF_df.unc/SOF_df.flux)**2 + (Spitz_uncnew/Spitz_fluxnew)**2)
SOFG_diff=(SOF_df.flux/G_fluxnew)
SOFG_diffavg=np.mean(SOFG_diff)
SOFG_diffunc = SOFG_diff * np.sqrt((SOF_df.unc/SOF_df.flux)**2 + (G_uncnew/G_fluxnew)**2)

one=np.ones(5)
fig, ax=plt.subplots()
ax.errorbar(SOF_df.wave, SOFSpitz_diff, yerr=SOFSpitz_diffunc, linewidth=1.0, color='b')
plt.title('Comparison of SOFIA and Spitzer Spectra')  
plt.xlabel('wavelength (microns)')
plt.ylabel('SOFIA Flux / Spitzer Flux')
ax.plot(np.arange(8,13), one, linewidth=1.0, color='g')
plt.xlim(9,12)
plt.show()

fig, ax= plt.subplots()
ax.errorbar(SOF_df.wave, SOFG_diff, yerr=SOFG_diffunc, linewidth=1.0, color='b')
plt.title('Comparison of SOFIA and Ground-based Spectra')
plt.xlabel('wavelength (microns)')
plt.ylabel('SOFIA Flux / Ground Flux')
ax.plot(np.arange(8,13), one, linewidth=1.0, color='g')
plt.xlim(9, 12)
plt.show()

#Calculating chi-squares

#chi_SOFSpitz=np.sum(((SOF_df.flux-Spitz_fluxnew)**2)/(SOF_df.unc**2))
#
#x_G=G_fluxnew-Spitz_fluxnew
#y_G=G_uncnew
#chi_GSpitz=np.sum(((x_G[~np.isnan(x_G)])**2)/((y_G[~np.isnan(y_G)])**2))

#Calculating chi-square of the ratio

#chi_SOFSpitzratio = np.sum((SOFSpitz_diff - 1)**2/(SOFSpitz_diffunc**2))
chi_SOFSpitzratio1=chisquare(SOFSpitz_diff[~np.isnan(SOFSpitz_diff)], f_exp=np.ones(139), axis=None)
chi_SOFSpitzratioAvg=chisquare(SOFSpitz_diff[~np.isnan(SOFSpitz_diff)], f_exp=(np.ones(139)*SOFSpitz_diffavg))

chi_SOFGratio1 = chisquare(SOFG_diff[~np.isnan(SOFG_diff)], f_exp=np.ones(139))
chi_SOFGratioAvg = chisquare(SOFG_diff[~np.isnan(SOFG_diff)], f_exp=(np.ones(139)*SOFG_diffavg))


#CUSTOM CHI-SQUARED VALUES
chiarr=[]
for x,y in zip(SOFSpitz_diff[~np.isnan(SOFSpitz_diff)], SOFSpitz_diffunc[~np.isnan(SOFSpitz_diffunc)]):
    chisq= (((x-1)**2)/(y**2))
    chiarr.append(chisq)

chisqSOFSpitz=np.sum(chiarr)

chiarr_A=[]
for x,y in zip(SOFSpitz_diff[~np.isnan(SOFSpitz_diff)], SOFSpitz_diffunc[~np.isnan(SOFSpitz_diffunc)]):
    chisq= (((x-SOFSpitz_diffavg)**2)/(y**2))
    chiarr_A.append(chisq)

chisqSOFSpitz_avg=np.sum(chiarr_A)

chiarr2=[]
for x,y in zip(SOFG_diff[~np.isnan(SOFG_diff)], SOFG_diffunc[~np.isnan(SOFG_diffunc)]):
    chisq= (((x-1)**2)/(y**2))
    chiarr2.append(chisq)
chisqSOFG=np.sum(chiarr2)

chiarr2_A=[]
for x,y in zip(SOFG_diff[~np.isnan(SOFG_diff)], SOFG_diffunc[~np.isnan(SOFG_diffunc)]):
    chisq= (((x-SOFG_diffavg)**2)/(y**2))
    chiarr2_A.append(chisq)
chisqSOFG_avg=np.sum(chiarr2_A)

#--------------------------------------------------------------------------------------------------------

#Creating dataframes for total spectral data

SOF_datatot={'wave': sofia_wave, 'flux':sofia_flux, 'unc': sofia_errtot}
SOF_dftot=pd.DataFrame(data=SOF_datatot)
SOF_dftot.dropna()

G_datatot_original={'wave': groundwave_original, 'flux': groundflux_original, 'unc': grounderr_original}
G_dftot_original=pd.DataFrame(data=G_datatot_original)
G_dftot_original.fillna(0)

G_datatot={'wave': groundwave, 'flux': groundflux, 'unc': grounderr}
G_dftot=pd.DataFrame(data=G_datatot)
G_dftot.dropna()

Spitz_datatot={'wave': spitzerwave, 'flux':spitzerflux, 'unc': spitzererr}
Spitz_dftot=pd.DataFrame(data=Spitz_datatot)
Spitz_dftot.dropna()

#Interpolate the Spitzer and ground-based spectra onto the SOFIA spectrum

Spitzinterpoltot=interpolate.interp1d(Spitz_dftot.wave, Spitz_dftot.flux, bounds_error=False)
Spitztot_wavenew=SOF_dftot.wave
Spitztot_fluxnew=Spitzinterpoltot(Spitztot_wavenew)

Spitzinterpoltot_unc=interpolate.interp1d(Spitz_dftot.wave, Spitz_dftot.unc, bounds_error=False)
Spitztot_uncnew=Spitzinterpoltot_unc(Spitztot_wavenew)

G_original_interpoltot=interpolate.interp1d(G_dftot_original.wave, G_dftot_original.flux, bounds_error=False)
Gtot_original_wavenew=SOF_dftot.wave
Gtot_original_fluxnew=G_original_interpoltot(Gtot_original_wavenew)

G_original_interpoltot_unc=interpolate.interp1d(G_dftot_original.wave, G_dftot_original.unc, bounds_error=False)
Gtot_original_uncnew=G_original_interpoltot_unc(Gtot_original_wavenew)

Ginterpoltot=interpolate.interp1d(G_dftot.wave, G_dftot.flux, bounds_error=False)
Gtot_wavenew=SOF_dftot.wave
Gtot_fluxnew=Ginterpoltot(Gtot_wavenew)

Ginterpoltot_unc=interpolate.interp1d(G_dftot.wave, G_dftot.unc, bounds_error=False)
Gtot_uncnew=Ginterpoltot_unc(Gtot_wavenew)

#PLOTS

fig, ax = plt.subplots()
ax.errorbar(SOF_dftot.wave, SOF_dftot.flux, SOF_dftot.unc, linewidth=1.0, color='b')
ax.errorbar(Gtot_wavenew, Gtot_fluxnew, Gtot_uncnew, linewidth=1.0, color='r')
ax.errorbar(Spitztot_wavenew, Spitztot_fluxnew, Spitztot_uncnew, linewidth=1.0, color='g')
plt.xlabel('wavelength (microns)')
plt.ylabel('average flux (Jy)')
plt.title('BD +20 307 Spectra (Spitzer and ground-based spectra linearly interpolated)')
blue_patch = mpatches.Patch(color='blue', label='$SOFIA$ $Spectrum$')
red_patch = mpatches.Patch(color='red', label='$Ground-based$ $Spectrum$')
green_patch = mpatches.Patch(color='green', label='$Spitzer$ $Spectrum$')
ax.legend(handles=[blue_patch, red_patch, green_patch], loc='upper right', fontsize='small')
plt.show()

SOFSpitztot_diff=(SOF_dftot.flux/Spitztot_fluxnew)
SOFSpitztot_diffAvg=np.mean(SOFSpitztot_diff)
SOFSpitztot_diffunc = SOFSpitztot_diff * np.sqrt((SOF_dftot.unc/SOF_dftot.flux)**2 + (Spitztot_uncnew/Spitztot_fluxnew)**2)

SOFGtot_diff=(SOF_dftot.flux/Gtot_fluxnew)
SOFGtot_diffAvg=np.mean(SOFGtot_diff)
SOFGtot_diffunc = SOFGtot_diff * np.sqrt((SOF_dftot.unc/SOF_dftot.flux)**2 + (Gtot_uncnew/Gtot_fluxnew)**2)

SpitzGoriginaltot_diff=(Spitztot_fluxnew/Gtot_original_fluxnew)
SpitzGoriginaltot_diffAvg=np.mean(SpitzGoriginaltot_diff)
SpitzGoriginaltot_diffunc=SpitzGoriginaltot_diff * np.sqrt((Spitztot_uncnew/Spitztot_fluxnew)**2 +(Gtot_original_uncnew/Gtot_original_fluxnew)**2)

SpitzGtot_diff=(Spitztot_fluxnew/Gtot_fluxnew)
SpitzGtot_diffAvg=np.mean(SpitzGtot_diff)
SpitzGtot_diffunc=SpitzGtot_diff * np.sqrt((Spitztot_uncnew/Spitztot_fluxnew)**2 +(Gtot_uncnew/Gtot_fluxnew)**2)


y=np.ones(8)
fig, ax=plt.subplots()
ax.errorbar(SOF_dftot.wave, SOFSpitztot_diff, yerr=SOFSpitztot_diffunc, linewidth=0.8, color='g')
plt.title(r'Comparison of SOFIA and Spitzer Spectra')  
plt.xlabel(r'wavelength ($\mu$m)')
plt.ylabel(r'SOFIA Flux / Spitzer Flux')
plt.xlim(8,14)
ax.plot(np.arange(7,15), y, '-')
#plt.savefig('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIASpitz_Diff.png', dpi=1000)
plt.show()

#Smoothing the above plot:

f=interpolate.interp1d(SOF_dftot.wave, SOFSpitztot_diff, kind='nearest')
g=interpolate.interp1d(SOF_dftot.wave, SOFSpitztot_diffunc, kind='nearest')
wave_smooth = SOF_dftot.wave[0::4]

fig, ax=plt.subplots()
ax.errorbar(wave_smooth, f(wave_smooth), yerr=g(wave_smooth), linewidth=1.5, color='g')
plt.title(r'Comparison of SOFIA and Spitzer Spectra')  
plt.xlabel(r'wavelength ($\mu$m)')
plt.ylabel(r'SOFIA Flux / Spitzer Flux')
plt.xlim(8,14)
ax.plot(np.arange(7,15), y, '-')
#plt.savefig('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIASpitz_DiffSmooth.png', dpi=1000)
plt.show()

fig,ax=plt.subplots()
ax.errorbar(SOF_dftot.wave, SOFGtot_diff, yerr=SOFGtot_diffunc, linewidth=0.8, color='g')
plt.title(r'Comparison of SOFIA and Ground-based Spectra')
plt.xlabel(r'wavelength ($\mu$m)')
plt.ylabel(r'SOFIA Flux / Keck-Gemini Flux')
plt.xlim(8,14)
ax.plot(np.arange(7,15), y, '-')
#plt.savefig('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIAGround_Diff.png', dpi=1000)
plt.show()

#Smoothing the above plot:
    
f_1=interpolate.interp1d(SOF_dftot.wave, SOFGtot_diff, kind='nearest')
g_1=interpolate.interp1d(SOF_dftot.wave, SOFGtot_diffunc, kind='nearest')

fig,ax=plt.subplots()
ax.errorbar(wave_smooth, f_1(wave_smooth), yerr=g_1(wave_smooth), linewidth=1.5, color='g')
plt.title(r'Comparison of SOFIA and Ground-based Spectra')
plt.xlabel(r'wavelength ($\mu$m)')
plt.ylabel(r'SOFIA Flux / Keck-Gemini Flux')
plt.xlim(8,14)
ax.plot(np.arange(7,15), y, '-')
#plt.savefig('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIAGround_DiffSmooth.png', dpi=1000)
plt.show()

fig,ax=plt.subplots()
ax.errorbar(SOF_dftot.wave, SpitzGoriginaltot_diff, yerr=SpitzGoriginaltot_diffunc, linewidth=0.8, color='g')
plt.title(r'Comparison of Spitzer and Original Ground-based Spectra')
plt.xlabel(r'wavelength ($\mu$m)')
plt.ylabel(r'Spitzer Flux / Keck-Gemini Flux')
plt.xlim(8,14)
ax.plot(np.arange(7,15), y, '-')
#plt.savefig('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIAGround_Diff.png', dpi=1000)
plt.show()

fig,ax=plt.subplots()
ax.errorbar(SOF_dftot.wave, SpitzGtot_diff, yerr=SpitzGtot_diffunc, linewidth=0.8, color='g')
plt.title(r'Comparison of Spitzer and Ground-based Spectra')
plt.xlabel(r'wavelength ($\mu$m)')
plt.ylabel(r'Spitzer Flux / Keck-Gemini Flux')
plt.xlim(8,14)
ax.plot(np.arange(7,15), y, '-')
#plt.savefig('/Users/maggiethompson/Desktop/CarnegieDTM/SOFIA_Spectra/SOFIAGround_Diff.png', dpi=1000)
plt.show()


#Calculating chi-squared values:

#chi_SOFSpitztot=np.sum(((SOF_dftot.flux-Spitztot_fluxnew)**2)/(SOF_dftot.unc**2))
#
#x_Gtot=Gtot_fluxnew-Spitztot_fluxnew
#y_Gtot=Gtot_uncnew
#chi_GSpitztot=np.sum(((x_Gtot[~np.isnan(x_Gtot)])**2)/((y_Gtot[~np.isnan(y_Gtot)])**2))

chi_SOFSpitz_ratiotot=chisquare(SOFSpitztot_diff[~np.isnan(SOFSpitztot_diff)], f_exp=np.ones(239))
chi_SOFSpitz_ratiototAvg=chisquare(SOFSpitztot_diff[~np.isnan(SOFSpitztot_diff)], f_exp=(np.ones(239)*SOFSpitztot_diffAvg))

chi_SOFG_ratiotot = chisquare(SOFGtot_diff[~np.isnan(SOFGtot_diff)], f_exp=np.ones(215))
chi_SOFG_ratiototAvg=chisquare(SOFGtot_diff[~np.isnan(SOFGtot_diff)], f_exp=(np.ones(215)*SOFGtot_diffAvg))

chiarrtot=[]
for x,y in zip(SOFSpitztot_diff[~np.isnan(SOFSpitztot_diff)], SOFSpitztot_diffunc[~np.isnan(SOFSpitztot_diffunc)]):
    chisq= (((x-1)**2)/(y**2))
    chiarrtot.append(chisq)

chisqSOFSpitzTot=np.sum(chiarrtot)

chiarrtot_A=[]
for x,y in zip(SOFSpitztot_diff[~np.isnan(SOFSpitztot_diff)], SOFSpitztot_diffunc[~np.isnan(SOFSpitztot_diffunc)]):
    chisq= (((x-SOFSpitztot_diffAvg)**2)/(y**2))
    chiarrtot_A.append(chisq)

chisqSOFSpitzTot_Avg=np.sum(chiarrtot_A)

chiarr2tot=[]
for x,y in zip(SOFGtot_diff[~np.isnan(SOFGtot_diff)], SOFGtot_diffunc[~np.isnan(SOFGtot_diffunc)]):
    chisq= (((x-1)**2)/(y**2))
    chiarr2tot.append(chisq)
    
chisqSOFGTot=np.sum(chiarr2tot)

chiarr2tot_A=[]
for x,y in zip(SOFGtot_diff[~np.isnan(SOFGtot_diff)], SOFGtot_diffunc[~np.isnan(SOFGtot_diffunc)]):
    chisq= (((x-SOFGtot_diffAvg)**2)/(y**2))
    chiarr2tot_A.append(chisq)
    
chisqSOFGTot_Avg=np.sum(chiarr2tot_A)

chiarrtot_test=[]
for x,y in zip(SOFSpitztot_diff[~np.isnan(SOFSpitztot_diff)], SOFGtot_diffunc[~np.isnan(SOFGtot_diffunc)]):
    chisq=(((x-SOFGtot_diffAvg)**2)/(y**2))
    chiarrtot_test.append(chisq)
chisqSOFSpitzTest=np.sum(chiarrtot_test)


