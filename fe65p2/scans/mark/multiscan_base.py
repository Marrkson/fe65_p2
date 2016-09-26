from fe65p2.scans.threshold_scan import ThresholdScan
from fe65p2.scans.timewalk_scan import TimewalkScan
from fe65p2.scan_base import ScanBase
#import fe65p2.scans.manualplot as manualplot
import numpy as np
from bokeh.plotting import figure, show, output_file, save, vplot, hplot
from glob import glob
from fe65p2 import fe65p2
import tables as tb
import yaml
import os

def multirowthreshscan(folder,outfolder):
    print folder
    a =  folder.rfind("/")
    out_var=outfolder + "/" + folder[a:]
    if not os.path.exists(outfolder + "/" + folder[a:]):
        os.makedirs(outfolder + "/" + folder[a:])
    folder = glob(folder + "/*")
    thresholdscan = ThresholdScan()
    for folders in folder:
        b = folders.rfind("/")
        print folders[b:]
        if not os.path.exists(out_var + str(folders[b:])):
            os.makedirs(out_var + str(folders[b:]))
        folders = folders + "/*.h5"
        folders = glob(folders)
        for files in folders:
            with tb.open_file(files, 'r') as in_file_h5:
                kwargs = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
                dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
            print outfolder + files[a:-3] + '_threshold'
            thresholdscan.output_filename = outfolder + files[a:-3] + '_threshold'
            kwargs['stop_pixel_count']
            thresholdscan.start(mask_steps=4, repeat_command=100, columns= 16*[True], scan_range=[0, 0.2, 0.005],#0.005
                                vthin1Dac=dac_status['vthin1Dac'] + 3, vthin2Dac=dac_status['vthin2Dac'], preCompVbnDac=dac_status['preCompVbnDac'], PrmpVbpDac=dac_status['PrmpVbpDac'],
                                mask_filename=files)
            thresholdscan.analyze()
    return out_var

def multirowtimescan(folder):
    print folder
    a = folder.rfind("/")
    folder = glob(folder + "/*")
    timewalkscan = TimewalkScan()
    print folder
    for folders in folder:
        i=int(folders[-2:])
        randoms=()
        for n in range(9):
           randoms=np.append(randoms,np.random.randint(0,63))
        folders = folders + "/*_threshold.h5"
        folders = glob(folders)
        for files in folders:
            with tb.open_file(files, 'r') as in_file_h5:
                kwargs = yaml.load(in_file_h5.root.meta_data.attrs.kwargs)
                dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
                Thresh_gauss = in_file_h5.root.Thresh_results.Threshold_pure.attrs.fitdata_thresh
            timewalkscan.output_filename = files[:-3] + '_timewalk'
            scanrange=[abs(Thresh_gauss['mu']-0.04),Thresh_gauss['mu']+0.2,abs(Thresh_gauss['mu']-0.04-Thresh_gauss['mu']-0.2)/40]

            pixelliste=[(2+8*i,randoms[0]),(2+8*i,randoms[1]),(3+8*i,randoms[2]),
                        (4+8*i,randoms[3]),(5+8*i,randoms[4]),(6+8*i,randoms[5]),
                        (7+8*i,randoms[6]),]
            print pixelliste
            timewalkscan.start(pix_list = pixelliste, mask_steps = kwargs['mask_steps'], repeat_command = 1001, columns = [True] * 16,
            scan_range = scanrange , vthin1Dac = dac_status['vthin1Dac'] +20, preCompVbnDac = dac_status['preCompVbnDac'],
                               PrmpVbpDac= dac_status['PrmpVbpDac'], vthin2Dac = dac_status['vthin2Dac'], mask_filename = files)
            timewalkscan.tdc_table(len(np.arange(scanrange[0],scanrange[1], scanrange[2]))+3)

if __name__ == "__main__":
    multirowthreshscan('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/Noisescans/preCompVbn', '/media/mark/1TB/Scanresults/output_data/new_chips/chip1/external')
    pass