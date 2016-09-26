
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as  plotting
import time
import fe65p2.analysis as analysis
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, show, vplot, hplot, save
from progressbar import ProgressBar
from basil.dut import Dut
import os

local_configuration = {
    "mask_steps": 1,
    "vthin1Dac": 22,
    "PrmpVbpDac": 80,
    "preCompVbnDac" : 110,
    "columns" : [False] * 0 + [True] * 2 + [False] * 14,
    "clock_en1" : True,#True
    "clock_en2" : True,#True
    "trigger_en" : True,#False
    "offset" : 5,
    "pixels" : 1, #64 #512
    "measure_direction" : True, #True high to low #False low to high
    "mask_filename": '/media/mark/1TB/Scanresults/output_data/new_chips/chip1/20160906_120820_noise_scan_standard.h5'
}

class NoiseThreshScan(ScanBase):
    scan_id = "noise_scan_thresh"


    def scan(self, clock_en1=True, pixels=512, clock_en2=True, trigger_en=True, measure_direction=True,  offset=15, mask_steps=4, PrmpVbpDac=80, vthin2Dac=0, columns = [True] * 16, vthin1Dac = 80, preCompVbnDac = 50, mask_filename='', **kwargs):

        '''Scan loop
        Parameters
        ----------
        mask : int
            Number of mask steps.
        repeat : int
            Number of injections.
        '''
        inj_factor = 1.0
        INJ_LO = 0.0
        #try:
            #dut = Dut(ScanBase.get_basil_dir(self)+'/examples/lab_devices/agilent33250a_pyserial.yaml')
            #dut.init()
            #logging.info('Connected to '+str(dut['Pulser'].get_info()))
        #except RuntimeError:
            #INJ_LO = 0.0#0.2
            #inj_factor = 2.0
            #logging.info('External injector not connected. Switch to internal one')
            #self.dut['INJ_LO'].set_voltage(INJ_LO, unit='V')
        offset=offset-1
        vthin1Dac=vthin1Dac+1

        self.dut['global_conf']['PrmpVbpDac'] = 80
        self.dut['global_conf']['vthin1Dac'] = 255
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['vffDac'] = 24
        self.dut['global_conf']['PrmpVbnFolDac'] = 51
        self.dut['global_conf']['vbnLccDac'] = 1
        self.dut['global_conf']['compVbnDac'] = 25
        self.dut['global_conf']['preCompVbnDac'] = 50

        self.dut.write_global()
        self.dut['control']['RESET'] = 0b01
        self.dut['control']['DISABLE_LD'] = 0
        self.dut['control']['PIX_D_CONF'] = 0
        self.dut['control'].write()

        self.dut['control']['CLK_OUT_GATE'] = 1
        self.dut['control']['CLK_BX_GATE'] = 1
        self.dut['control'].write()
        time.sleep(0.1)

        self.dut['control']['RESET'] = 0b11
        self.dut['control'].write()

        self.dut['global_conf']['OneSr'] = 1

        self.dut['global_conf']['TestHit'] = 0
        self.dut['global_conf']['SignLd'] = 0
        self.dut['global_conf']['InjEnLd'] = 0
        self.dut['global_conf']['TDacLd'] = 0
        self.dut['global_conf']['PixConfLd'] = 0
        self.dut.write_global()

        #self.dut['global_conf']['OneSr'] = 0  #all multi columns in parallel
        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray([True] * 16) #(columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray([True] * 16)
        self.dut.write_global()


        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel()
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut.write_global()
        self.dut['global_conf']['InjEnLd'] = 0

        mask_en = np.full([64,64], False, dtype = np.bool)
        mask_tdac = np.full([64,64], 16, dtype = np.uint8)
        ###
        if pixels>1 and pixels<=64:
            mask_en[1:2, :] = True
        ###
        if pixels==1:
            mask_en[1][1]=True

        if mask_filename:
            logging.info('Using pixel mask from file: %s', mask_filename)

            with tb.open_file(mask_filename, 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                if pixels>64:
                    mask_en = in_file_h5.root.scan_results.en_mask[:]

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)
        self.dut.write_global()

        self.dut['global_conf']['OneSr'] = 0
        self.dut.write_global()

        self.dut['trigger'].set_delay(10000) #trigger for no injection 10000
        self.dut['trigger'].set_width(16)#16
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(False)

        logging.debug('Configure TDC')
        self.dut['tdc']['RESET'] = True
        self.dut['tdc']['EN_TRIGGER_DIST'] = True
        self.dut['tdc']['ENABLE_EXTERN'] = False
        self.dut['tdc']['EN_ARMING'] = False
        self.dut['tdc']['EN_INVERT_TRIGGER'] = False
        self.dut['tdc']['EN_INVERT_TDC'] = False
        self.dut['tdc']['EN_WRITE_TIMESTAMP'] = True

        lmask = [1] + ( [0] * (mask_steps-1) )
        lmask = lmask * ( (64 * 64) / mask_steps  + 1 )
        lmask = lmask[:64*64]
        ranges=np.arange(0,(vthin1Dac-offset),1)
        n=0
        for ni in ranges:
            time.sleep(0.5)
            bv_mask = bitarray.bitarray(lmask)
            if measure_direction:
                vthin1Dac1 = vthin1Dac - n
            else:
                vthin1Dac1 = n + offset
            with self.readout(scan_param_id = vthin1Dac1):#vthin1Dac-n):
                logging.info('Scan Parameter: %f (%d of %d)', vthin1Dac1, n+1, vthin1Dac-offset)
                pbar = ProgressBar(maxval=mask_steps).start()

                self.dut['global_conf']['vthin1Dac'] = 255
                self.dut['global_conf']['preCompVbnDac'] = 50
                self.dut['global_conf']['vthin2Dac'] = 0
                self.dut['global_conf']['PrmpVbpDac'] = 80
                self.dut.write_global()
                time.sleep(0.1)

                self.dut['pixel_conf'][:]  = bv_mask
                self.dut.write_pixel_col()
                self.dut['global_conf']['InjEnLd'] = 0#1
                #self.dut['global_conf']['PixConfLd'] = 0b11
                self.dut.write_global()

                bv_mask[1:] = bv_mask[0:-1]
                bv_mask[0] = 0
                self.dut['global_conf']['vthin1Dac'] = vthin1Dac1
                self.dut['global_conf']['preCompVbnDac'] = preCompVbnDac
                self.dut['global_conf']['vthin2Dac'] = vthin2Dac
                self.dut['global_conf']['PrmpVbpDac'] = PrmpVbpDac
                self.dut.write_global()
                time.sleep(0.1)

                #while not self.dut['inj'].is_done():
                    #pass

                if trigger_en==True:
                    self.dut['trigger'].set_repeat(0)
                if clock_en1==False:
                    self.dut['control']['CLK_BX_GATE'] = 0
                    self.dut['control'].write()
                if clock_en2==False:
                    self.dut['control']['CLK_OUT_GATE'] = 0
                    self.dut['control'].write()
                if trigger_en == True:
                    self.dut['trigger'].start()

                self.dut['tdc'].ENABLE = True

                time.sleep(5)#10

                self.dut['tdc'].ENABLE = False

                #n=0
                #if trigger_en == True:
                    #while not self.dut['trigger'].is_done():
                        #time.sleep(1)
                        #n=n+1
                        #print self.dut['trigger'].is_done() , n, "sekunden"

                if clock_en1==False:
                    self.dut['control']['CLK_BX_GATE'] = 1
                    self.dut['control'].write()
                if clock_en2==False:
                    self.dut['control']['CLK_OUT_GATE'] = 1
                    self.dut['control'].write()
                #while not self.dut['trigger'].is_done():
                    #pass
                n=n+1


        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Masks')
        self.h5_file.createCArray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.createCArray(scan_results, 'en_mask', obj=mask_en)

    def analyze(self, trigger, clock1, clock2, measuredirection,ranges):
        h5_filename = self.output_filename +'.h5'
        get_bin = lambda x, n: format(x, 'b').zfill(n)
        ranges1 = ranges[1] #+ 1
        ranges0= ranges[0] #- 1
        attrs_tdc = {}
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            en_mask = in_file_h5.root.scan_results.en_mask[:]
            #if trigger==1:
                #hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
                #in_file_h5.createTable(in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
                #hit_noise=np.zeros(ranges[1]-ranges[0]+2)
                #for entries in hit_data:
                    #hit_noise[entries['scan_param_id']-ranges[0]+1]=hit_noise[entries['scan_param_id']-ranges[0]+1]+1
            new_meta = []
            new_meta_trigger=[]
            if len(meta_data):
                for elements in meta_data:
                    if elements[7] > 0:
                        if len(new_meta) < 1:
                            new_meta = elements
                        else:
                            new_meta = np.vstack((new_meta, elements))
                    if elements[7]==0:
                        if len(new_meta_trigger) < 1:
                            new_meta_trigger = elements
                        else:
                            new_meta_trigger = np.vstack((new_meta_trigger, elements))
                param = []
                noise = []
                TOT = []
                TOT_sigma = []
                if len(new_meta):
                    param = np.arange(ranges0, ranges1)
                    for numbers in param:
                        noise_per_param = 0
                        TOT_per_param=0
                        TOT_sigma_per_param = 0
                        for entries in new_meta:
                            if numbers == entries['scan_param_id']:
                                for raw_entries in raw_data[entries['index_start']:entries['index_stop']]:
                                    if int(get_bin(int(raw_entries), 32)[1])==1:
                                        tdc_data = raw_entries & 0xFFF  # take last 12 bit
                                        noise_per_param = noise_per_param + 1
                                        TOT_per_param = TOT_per_param + tdc_data
                                        TOT_sigma_per_param=TOT_sigma_per_param+tdc_data*tdc_data


                        noise = np.append(noise, noise_per_param/5.0)
                        if noise_per_param/5.0==0:
                            TOT = np.append(TOT, 0)
                            TOT_sigma = np.append(TOT_sigma, 0)
                        else:
                            TOT = np.append(TOT,(TOT_per_param/noise_per_param) * 1.5625)
                            TOT_sigma=np.append(TOT_sigma,np.sqrt((TOT_sigma_per_param/noise_per_param )* 1.5625))
                        logging.info('Calculated Param: ' + str(numbers))
                en_mask1 = np.array(en_mask, dtype=bool)
                pixels = np.count_nonzero(en_mask1)
            attrs_tdc['trigger']=trigger
            attrs_tdc['clock1']=clock1
            attrs_tdc['clock2'] = clock2
            attrs_tdc['direction']=measuredirection
            attrs_tdc['pixels']=pixels
            avg_tab = np.rec.fromarrays([param, noise, TOT, TOT_sigma], dtype=[('vthin1Dac', float), ('noise', float), ('TOT_ns', float), ('err_TOT_ns', float)]) #, noise2, hit_noise     , ('noise2', int), ('hit_noise', int)
            tdc_data=in_file_h5.createTable(in_file_h5.root, 'tdc_data', avg_tab, filters=self.filter_tables)
            tdc_data.attrs.clock_trigger = attrs_tdc
        #for i in range(len(param)):
            #print param[i], noise[i], TOT[i], TOT_sigma[i]

        fig, ax1 = plt.subplots()
        plt.title("Trigger: " + str(trigger) + " CLK_BX_GATE: " + str(clock1) + " CLK_OUT_GATE: " + str(clock2) + " high to low: " + str(measuredirection) + " Pixels: " + str(pixels))
        ax1.errorbar(param, noise, fmt='ro', yerr=np.sqrt(noise))
        ax1.set_xlabel('global threshold Dac')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('noise [1/s]')
        ax1.set_yscale('symlog')
        for tl in ax1.get_yticklabels():
            tl.set_color('r')

        ax2 = ax1.twinx()
        ax2.errorbar(param, TOT, fmt='bo', yerr=TOT_sigma)
        ax2.set_ylabel('noise length [ns]')
        ax2.set_yscale('symlog')
        for tl in ax2.get_yticklabels():
            tl.set_color('b')
        plt.draw()
        #plt.show()
        plt.savefig(h5_filename[:-3]+".png")
        plt.close()



        #plt.title("Trigger: " + str(trigger) + " CLK_BX_GATE: " + str(clock1) + " CLK_OUT_GATE: " + str(clock2) + " high to low: " + str(measuredirection) + " Pixels: " + str(pixels))
        #plt.plot(param, noise, linestyle='--', marker='o')
        #plt.errorbar(param, noise, fmt='ro', yerr=np.sqrt(noise))
        ##plt.plot(param, noise2, linestyle='None', marker='o')
        ##plt.ylim([0.1,10000000000])
        #plt.xlim([np.min(param)-1,np.max(param)+1])
        #plt.xlabel('global threshold Dac')
        #plt.ylabel('noise [1/s]')
        #plt.yscale('symlog')
        #plt.draw()
        #plt.savefig(h5_filename[:-3]+".png")
        #plt.close()


if __name__ == "__main__":

    scan = NoiseThreshScan()
    scan.start(**local_configuration)
    scan.analyze(local_configuration['trigger_en'], local_configuration['clock_en1'], local_configuration['clock_en2'],local_configuration['measure_direction'],[local_configuration['offset'],local_configuration['vthin1Dac']])
    #scan.analyze('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/clock_influence/high_to_low_512/trigger_0_clock1_1_clock2_1.h5',1, 1, 1, 1, [3,25])