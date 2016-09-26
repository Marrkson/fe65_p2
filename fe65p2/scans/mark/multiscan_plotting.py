import numpy as np
import yaml
import matplotlib.pyplot as plt
import tables as tb
import logging
from glob import glob
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import fe65p2.analysis as analysis
import os
import shutil



def plot_thresh(directory,savepoint,parameter): #preamp
    directory1=directory
    directory=directory+'/*_threshold.h5'
    directory=glob(directory)
    vthin1Dac_array=()
    power_status_array=()
    Threshold_array=()
    PrmpVbpDac_array=()
    Threshold_sigma_array=()
    for files in directory:
        with tb.open_file(files, 'r') as in_file_h5:
            vthin1Dac= yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
            power_status = yaml.load(in_file_h5.root.meta_data.attrs.power_status)
            Threshold= in_file_h5.root.Thresh_results.Threshold_pure.attrs.fitdata_thresh
            vthin1Dac_array = np.append(vthin1Dac_array,vthin1Dac['vthin1Dac'])
            power_status_array = np.append(power_status_array,power_status['VDDA[mA]'])
            Threshold_array = np.append(Threshold_array,Threshold['mu'])
            Threshold_sigma_array = np.append(Threshold_sigma_array, Threshold['sigma'])
            PrmpVbpDac_array = np.append(PrmpVbpDac_array,vthin1Dac[parameter])
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=(par2),
                                        offset=(offset, 0))

    par2.axis["right"].toggle(all=True)

    host.set_xlabel(parameter)
    host.set_ylabel("VDDA [mA] ")
    host.set_ylim([18, 38])
    par1.set_ylabel("Threshold [e-]")
    par1.set_ylim([0,1100])
    par2.set_ylabel("vthin1Dac")
    par2.set_ylim([0, 100])

    p1, = host.plot(PrmpVbpDac_array, power_status_array,'ro')
    host.errorbar(PrmpVbpDac_array, power_status_array, fmt='ro', yerr=0.05)
    p2, = par1.plot(PrmpVbpDac_array, Threshold_array * 1000*analysis.cap_fac(),'bv')
    par1.errorbar(PrmpVbpDac_array, Threshold_array*1000*analysis.cap_fac(), fmt='bv', yerr=(Threshold_sigma_array*1000*analysis.cap_fac()))#7.6
    p3, = par2.plot(PrmpVbpDac_array, vthin1Dac_array,'gs')
    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())

    plt.draw()
    a = directory1.rfind("/")
    plt.savefig(savepoint)
    plt.close()

def plot_noise(directory,savepoint,parameter): #precomp
    directory = directory + '/*_threshold.h5'
    directory=glob(directory)
    vthin1Dac_array=()
    power_status_array=()
    noise_array=()
    PrmpVbpDac_array=()
    noise_sigma_array=()
    for files in directory:
        with tb.open_file(files, 'r') as in_file_h5:
            vthin1Dac= yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
            power_status = yaml.load(in_file_h5.root.meta_data.attrs.power_status)
            noise= in_file_h5.root.Noise_results.Noise_pure.attrs.fitdata_noise
            vthin1Dac_array = np.append(vthin1Dac_array,vthin1Dac['vthin1Dac'])
            power_status_array = np.append(power_status_array,power_status['VDDA[mA]'])
            noise_array = np.append(noise_array,noise['mu'])
            noise_sigma_array = np.append(noise_sigma_array, noise['sigma'])
            PrmpVbpDac_array = np.append(PrmpVbpDac_array,vthin1Dac[parameter])
            mask_en = in_file_h5.root.scan_results.en_mask[:]
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    host.set_xlabel(parameter)
    host.set_ylabel("Current per Pixel[$\mu$A]")
    host.set_ylim([0,20])
    par1.set_ylabel("Noise [e-]")
    par1.set_ylim([20,200])

    if np.any(mask_en[17][17:21]):
        factor=4096
    else:
        factor=4096-512
    p1, = host.plot(PrmpVbpDac_array, power_status_array/factor*1000,'ro')
    host.errorbar(PrmpVbpDac_array, power_status_array/factor*1000, fmt='ro', yerr=power_status_array/factor*1000*0.2)
    p2, = par1.plot(PrmpVbpDac_array, noise_array * 1000 *analysis.cap_fac(),'bv')
    par1.errorbar(PrmpVbpDac_array, noise_array*1000 * analysis.cap_fac(), fmt='bv', yerr=(noise_sigma_array*1000* analysis.cap_fac()))
    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    plt.savefig(savepoint)
    plt.close()

def plot_deadpixels(folder,savefile,parameter,chip): #vthin2Dac
    folder = glob(folder+'/*_threshold.h5')
    deadpixels=()
    untuned=()
    Precomp=()
    for files in folder:
        with tb.open_file(files, 'r') as in_file_h5:
            dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
            t_dac = in_file_h5.root.scan_results.tdac_mask[:]
            en_mask = in_file_h5.root.scan_results.en_mask[:]
            Threshold= in_file_h5.root.Thresh_results.Threshold_pure[:]
            hist, edges = np.histogram(Threshold, density=False, bins=50)
            deadpixels=np.append(deadpixels, hist[0])
            Precomp = np.append(Precomp, dac_status[parameter])
        shape = en_mask.shape
        ges = 1
        for i in range(2):
            ges = ges * shape[i]
        T_Dac_pure = ()
        t_dac = t_dac.reshape(ges)
        en_mask = en_mask.reshape(ges)
        for i in range(ges):
            if (str(en_mask[i]) == 'True'):
                T_Dac_pure = np.append(T_Dac_pure, t_dac[i])
        T_Dac_pure = T_Dac_pure.astype(int)
        T_Dac_hist_y = np.bincount(T_Dac_pure)
        untuned = np.append(untuned, T_Dac_hist_y[chip])
    plt.subplots_adjust(right=0.75)
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    par1 = host.twinx()
    host.set_xlabel(parameter)
    host.set_ylabel("Untuned pixels")
    par1.set_ylabel("Not responding pixels")
    host.set_ylim([0,500])
    par1.set_ylim([0,500])
    p1, = host.plot(Precomp, untuned, 'ro')
    p2, = par1.plot(Precomp,deadpixels, 'bv')
    host.legend()
    host.axis["left"].label.set_color(p1.get_color())
    #host.errorbar(Precomp, untuned, fmt='ro', yerr=(np.sqrt(untuned)))
    par1.axis["right"].label.set_color(p2.get_color())
    #par1.errorbar(Precomp, deadpixels, fmt='bv', yerr=(np.sqrt(deadpixels)))
    plt.savefig(savefile)
    plt.close()

def timewalk_reevaluated(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        try:
            tdc_data = in_file_h5.root.tdc_data[:]
            td_threshold=in_file_h5.root.td_threshold[:]
        except RuntimeError:
            logging.info('tdc_data not present in file')
            return
        time_thresh = td_threshold['td_threshold']

        tot = tdc_data['tot_ns']
        delay = tdc_data['delay_ns']
        pixel_no = tdc_data['pixel_no']
        pulse = tdc_data['charge']
        hits = tdc_data['hits']
        pix, stop = np.unique(pixel_no, return_index=True)
        stop = np.sort(stop)

        stop = list(stop)
        stop.append(len(tot))
        in_time_threshold_array=[]
        for i in range(len(stop)-1):
            s1 = int(stop[i])
            s2 = int(stop[i+1])
            if time_thresh[i]==0:
                continue
            A, mu, sigma = analysis.fit_scurve(hits[s1:s2], pulse[s1:s2], np.max(hits[s1:s2]))
            for values in range(s1,s2):
                if pulse[values] >=5/4*mu:
                    s1=values
                    break
            if len(time_thresh)!=0:
                n=0
                delay_min=np.min(delay[s1:s2])
                for entries in pulse[s1:s2]:
                    if delay[s1+n]<=delay_min+25:
                        in_time_threshold_array=np.append(in_time_threshold_array,pulse[s1+n])
                        break
                    n=n+1
            else:
                logging.info('No fit possible. Some error occured.')

        return in_time_threshold_array




def plot_timewalk(folder,savefile,parameter): #precomp
    folder = glob(folder+'/*_threshold_timewalk.h5')
    threshold_all=()
    responds_all=()
    thresherror_all=()
    Preamp=()
    for files in folder:
        with tb.open_file(files, 'r') as in_file_h5:
            dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
            thresh= timewalk_reevaluated(files)
            #Threshold2=in_file_h5.root.td_threshold[:]
            #print "filename: ", files
            #print "Threshold1: ", thresh
            #print "Threshold2: ", Threshold2

            #responds=0
            #thresh=()
            #x=0
            #for i in Threshold:
                #x=x+1
                #if i[1]>0:
                    #responds=responds+1
                    #thresh=np.append(thresh,i[1])
            threshmean=np.mean(thresh)
            thresherror=np.std(thresh)
            responds_all=np.append(responds_all,len(thresh))
            Preamp = np.append(Preamp, dac_status[parameter])
            threshold_all=np.append(threshold_all,threshmean)
            thresherror_all = np.append(thresherror_all, thresherror)
    x=np.max(responds_all)
    x=int(x)
    plt.subplots_adjust(right=0.75)
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    par1 = host.twinx()
    host.set_xlabel(parameter)
    host.set_ylabel("In-time-threshold [e-]")
    par1.set_ylabel("Responding pixels max: " + str(x))
    host.set_ylim([300,1400])
    par1.set_ylim([0,70])
    p1, = host.plot(Preamp, threshold_all, 'ro')
    host.errorbar(Preamp, threshold_all, fmt='ro', yerr=(thresherror_all))
    if parameter=="PrmpVbpDac":
        width=2
    else:
        width=4
    par1.bar(Preamp,responds_all,width=width, color='b')
    host.legend()
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color('b')
    plt.savefig(savefile)
    plt.close()

def full_plot(folder,parameter):
    a = folder.rfind("/chip1/")
    b = folder.rfind("/new_chips/")
    if b==-1:
        age= "old_chips"
        chips_age=0
    else:
        age= "new_chips"
        chips_age=1
    if a==-1:
        chip="chip2"
    else:
        chip="chip1"
    a = folder.rfind("/internal/")
    if a==-1:
        injection="external"
    else:
        injection="internal"
    folder1=folder
    folder = glob(folder + "/*")
    name=folder1.rfind("/")
    for folders in folder:
        name2=folders.rfind("/")
        plot_thresh(folders,'/media/mark/1TB/Scanresults/output_data/' + age + '/' + chip  + '/Plots/new_Threshold/'+ folders[name2:] + "_" + injection + '.png',parameter)
        plot_noise(folders,'/media/mark/1TB/Scanresults/output_data/' + age + '/' + chip  + '/Plots/new_Noise/'+ folders[name2:] + "_"  + injection + '.png', parameter)
        plot_deadpixels(folders,'/media/mark/1TB/Scanresults/output_data/' + age + '/' + chip + '/Plots/new_Deadpixels/' + folders[name2:]+ "_"  + injection + '.png', parameter, chips_age)
        if injection=="external":
            plot_timewalk(folders, '/media/mark/1TB/Scanresults/output_data/' + age + '/' + chip + '/Plots/new_Timewalk/' + folders[name2:]+ "_"  + injection + '.png',parameter)

def find_file_and_copy(search_folder, paste_fodler, search_term, suffix):
    search_folder=search_folder + '/*.png'
    search_folder=glob(search_folder)
    for files in search_folder:
        print files
        if files.rfind(search_term) != -1:
            a=files.rfind('/')
            shutil.copy(files,  paste_fodler + files[a:-4] + '_' + suffix + '.png')


def sort_for_display(inputfolder, outputfolder):
    categories=['Feedback', 'LCC', 'Pwrdwn', 'W', 'vthin2Dac', 'Precompvbn', 'Prmpvbp']
    Folder_names=['new_Threshold', 'new_Noise', 'new_Timewalk', 'new_Deadpixels']
    Feedback=['vthin2Dac_03', 'vthin2Dac_06']
    LCC=['vthin2Dac_04', 'vthin2Dac_05']
    Pwrdwn=['preCompVbn_01','preCompVbn_02']
    W=['vthin2Dac_01', 'vthin2Dac_03']
    vthin2Dac=['vthin2Dac_06']
    Precompvbn=['preCompVbn_06']
    Prmpvbp=['PrmpVbpDac_06']
    for folders in categories:
        if not os.path.exists(outputfolder + '/' + folders):
            os.makedirs(outputfolder + '/' + folders)
            if folders=='Feedback':
                for i in Feedback:
                    for j in Folder_names:
                        find_file_and_copy(inputfolder + '/' + j, outputfolder + '/' + folders, i, j)
            if folders == 'LCC':
                for i in LCC:
                    for j in Folder_names:
                        find_file_and_copy(inputfolder + '/' + j, outputfolder + '/' + folders, i, j)
            if folders == 'Pwrdwn':
                for i in Pwrdwn:
                    for j in Folder_names:
                        find_file_and_copy(inputfolder + '/' + j, outputfolder + '/' + folders, i, j)
            if folders == 'W':
                for i in W:
                    for j in Folder_names:
                        find_file_and_copy(inputfolder + '/' + j, outputfolder + '/' + folders, i, j)
            if folders == 'vthin2Dac':
                for i in vthin2Dac:
                    for j in Folder_names:
                        find_file_and_copy(inputfolder + '/' + j, outputfolder + '/' + folders, i, j)
            if folders == 'Precompvbn':
                for i in Precompvbn:
                    for j in Folder_names:
                        find_file_and_copy(inputfolder + '/' + j, outputfolder + '/' + folders, i, j)
            if folders == 'Prmpvbp':
                for i in Prmpvbp:
                    for j in Folder_names:
                        find_file_and_copy(inputfolder + '/' + j, outputfolder + '/' + folders, i, j)







if __name__ == "__main__":
    #full_plot('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/external/preCompVbn', 'preCompVbnDac')
    #full_plot('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/external/PrmpVbpDac', 'PrmpVbpDac')
    #full_plot('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/external/vthin2Dacnew', 'vthin2Dac')
    #full_plot('/media/mark/1TB/Scanresults/output_data/old_chips/chip2/external/preCompVbn', 'preCompVbnDac')
    #full_plot('/media/mark/1TB/Scanresults/output_data/old_chips/chip2/external/PrmpVbpDac', 'PrmpVbpDac')
    #full_plot('/media/mark/1TB/Scanresults/output_data/old_chips/chip2/external/vthin2Dacnew', 'vthin2Dac')
    sort_for_display('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/Plots', '/home/mark/Dropbox/Bachelorarbeit/Presentation/figs/new_plots')

    pass
#params:
#preCompVbnDac
#vthin2Dac
#PrmpVbpDac
