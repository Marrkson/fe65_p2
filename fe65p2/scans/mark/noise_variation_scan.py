import numpy as np
import os
import yaml
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import tables as tb
from glob import glob
from noisethresh_scan import  NoiseThreshScan


def noise_scan_thresh_routine(config,ranges,scan):
    #scan =  NoiseThreshScan()
    maskfile='/media/mark/1TB/Scanresults/output_data/new_chips/chip1/20160906_120820_noise_scan_standard.h5'
    with tb.open_file(maskfile, 'r+') as in_file_h5:
        dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
    for rows in config:
        triggeren = rows[0]
        clocken1=rows[1]
        clocken2=rows[2]
        measuredirect=rows[3]
        pixel=rows[4]
        if measuredirect==True:
            measuredirect_str='high_to_low'
        else:
            measuredirect_str='low_to_high'
        folder = '/media/mark/1TB/Scanresults/output_data/new_chips/chip1/clock_influence/'+ str(measuredirect_str)+'_'+ str(pixel) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        output_filename=folder+'trigger_'+str(triggeren)+'_clock1_'+str(clocken1)+'_clock2_'+str(clocken2)
        scan.start(mask_steps=1, columns = [False] * 0 + [True] * 2 + [False] * 14, repeat_command=100, scan_range= [0.0, 0.2, 0.5],
                   mask_filename=maskfile,pixels=pixel,offset=ranges[0], vthin1Dac=ranges[1],
                   vthin2Dac=dac_status['vthin2Dac'], PrmpVbpDac=dac_status['PrmpVbpDac'], preCompVbnDac=dac_status['preCompVbnDac'],
                   clock_en1=clocken1, clock_en2=clocken2, trigger_en=triggeren, measure_direction=measuredirect)
        scan.analyze(triggeren, clocken1, clocken2, measuredirect,ranges)
        shutil.move(scan.output_filename+'.h5',output_filename+'.h5')
        shutil.move(scan.output_filename+'.log', output_filename + '.log')
        shutil.move(scan.output_filename+'.png', output_filename + '.png')
    return folder


def noise_sweep(folder,ranges):
    folder=folder+"/*.h5"
    folder=glob(folder)
    noise_all=np.zeros((len(folder),100))
    vthin1Dac_all=np.zeros((len(folder),100))
    attrs_all=[]
    counter=0
    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    for files in folder:
        with tb.open_file(files, 'r+') as in_file_h5:
            tdc_data = in_file_h5.root.tdc_data[:]
            vthin1Dac = tdc_data['vthin1Dac']
            noise = tdc_data['noise']
            tdc_attrs=in_file_h5.root.tdc_data.attrs.clock_trigger
        if len(attrs_all)<1:
            attrs_all=[tdc_attrs['trigger'],tdc_attrs['clock1'],tdc_attrs['clock2']]
        else:
            attrs_all=np.vstack((attrs_all,[tdc_attrs['trigger'],tdc_attrs['clock1'],tdc_attrs['clock2']]))
        noise_all[counter][:]=np.pad(noise,(0,100-len(noise)),'constant')
        vthin1Dac_all[counter][:] = np.pad(vthin1Dac,(0,100-len(vthin1Dac)),'constant')
        counter=counter+1
    color = iter(cm.rainbow(np.linspace(0, 1, len(attrs_all))))
    for i in range(len(attrs_all)):
        vthin1Dac = vthin1Dac_all[i][vthin1Dac_all[i] >0]
        noise = noise_all[i][:len(vthin1Dac)]
        c = next(color)
        if attrs_all[i][0]==True:
            trigger="on"
        else:
            trigger="off"
        if attrs_all[i][1]==True:
            clock1="on"
        else:
            clock1="off"
        if attrs_all[i][2]==True:
            clock2="on"
        else:
            clock2="off"
        legend="Trigger: " + trigger + " CLK_BX_GATE: " + clock1 + " CLK_OUT_GATE: " + clock2
        if len(vthin1Dac)<len(np.arange(ranges[0],ranges[1]+1,1)):
            counter=0
            for numbers in np.arange(ranges[0],ranges[1]+1,1):
                if numbers == vthin1Dac[0]:
                    break
                counter = counter + 1
            #print noise
            #print len(np.arange(ranges[0],ranges[1]+1,1)) - len(noise) - counter, len(np.arange(ranges[0],ranges[1]+1,1)) , len(noise), counter
            noise = np.pad(noise, (counter, len(np.arange(ranges[0],ranges[1]+1,1)) - len(noise) - counter), 'constant')
            vthin1Dac = np.arange(ranges[0],ranges[1]+1,1)

        plt.plot(vthin1Dac,noise, c=c, linestyle='--', marker='o',label=legend)
        plt.errorbar(vthin1Dac,noise, c=c, linestyle='None', marker='o', yerr=np.sqrt(noise))
        plt.xlabel('global threshold Dac')
        plt.ylabel('noise [1/s]')
        plt.legend(loc='upper left')

    plt.yscale('symlog')
    plt.draw()
    #plt.show()
    plt.savefig(folder[-1][:-3]+"_analysis.png")
    plt.close()


def whole_scan():
    ranges=[1,25]
    scan =  NoiseThreshScan()
    column_high_to_low = [[1, 1, 1, 1, 512], [0, 1, 1, 1, 512], [0, 0, 1, 1, 512], [0, 1, 0, 1, 512], [0, 0, 0, 1, 512]]
    folder=noise_scan_thresh_routine(column_high_to_low,ranges, scan)
    noise_sweep(folder, ranges)

    column_low_to_high = [[1, 1, 1, 0, 512], [0, 1, 1, 0, 512], [0, 0, 1, 0, 512], [0, 1, 0, 0, 512], [0, 0, 0, 0, 512]]
    folder=noise_scan_thresh_routine(column_low_to_high,ranges, scan)
    noise_sweep(folder, ranges)

    single_column_high_to_low = [[1, 1, 1, 1, 64], [0, 1, 1, 1, 64], [0, 0, 1, 1, 64], [0, 1, 0, 1, 64], [0, 0, 0, 1, 64]]
    folder=noise_scan_thresh_routine(single_column_high_to_low,ranges, scan)
    noise_sweep(folder,ranges)

    single_column_low_to_high = [[1, 1, 1, 0, 64], [0, 1, 1, 0, 64], [0, 0, 1, 0, 64], [0, 1, 0, 0, 64],[0, 0, 0, 0, 64]]
    folder=noise_scan_thresh_routine(single_column_low_to_high,ranges, scan)
    noise_sweep(folder,ranges)

    single_pix_high_to_low=[[1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1]]
    folder=noise_scan_thresh_routine(single_pix_high_to_low,ranges, scan)
    noise_sweep(folder, ranges)

    single_pix_low_to_high=[[1, 1, 1, 0, 1], [0, 1, 1, 0, 1], [0, 0, 1, 0, 1], [0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]
    folder=noise_scan_thresh_routine(single_pix_low_to_high,ranges, scan)
    noise_sweep(folder, ranges)


if __name__ == "__main__":
    #whole_scan()
    noise_sweep('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/clock_influence/12_09_2016/high_to_low_512',)
    pass