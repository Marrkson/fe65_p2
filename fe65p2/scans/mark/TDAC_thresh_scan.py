import numpy as np
import matplotlib.pyplot as plt
import tables as tb
from glob import glob
from fe65p2.scans.threshold_scan import ThresholdScan
from fe65p2.analysis import cap_fac

def TDAC_scan():
    scan = ThresholdScan()
    name=scan.output_filename
    for n in range(1):
        for i in range(1,32):
            print "n: ", n, " i: ", i
            scan.output_filename = name + '_' + str(n) + '_' + str("%02d" % i) + '_TDAC'
            scan.start(mask_steps=4, TDAC=i, repeat_command=100, PrmpVbpDac=80, vthin2Dac=0,
            columns = [False] * 4 + [True] * 2 + [False] * 12, scan_range = [0.0, 0.4, 0.01], vthin1Dac = 80,
            preCompVbnDac = 50, mask_filename='')
            scan.analyze()

def plot_thresh_as_TDAC(directory):
    directory = directory + '/*.h5'
    directory = glob(directory)
    TDAC = ()
    threshold = ()
    threshold_err = ()
    threshold_one=()
    for files in directory:
        with tb.open_file(files, 'r') as in_file_h5:
            TDAC = np.append(TDAC,in_file_h5.root.scan_results.tdac_mask[0][0])
            threshold=np.append(threshold, in_file_h5.root.Thresh_results.Threshold_pure.attrs.fitdata_thresh['mu'])
            threshold_err = np.append(threshold_err, in_file_h5.root.Thresh_results.Threshold_pure.attrs.fitdata_thresh['sigma'])
            threshold_one= np.append(threshold_one, in_file_h5.root.Thresh_results.Threshold_pure[0])
    plt.errorbar(TDAC,threshold*cap_fac()*1000, yerr=threshold_err*cap_fac()*1000, fmt='ro')
    plt.xlim([-1,32])
    plt.xlabel('TDAC')
    plt.ylabel('Threshold [e-]')
    plt.draw()
    plt.show()

if __name__ == "__main__":
    #TDAC_scan()
    plot_thresh_as_TDAC('/home/mark/Desktop/TDAC_plot_no16')
    pass