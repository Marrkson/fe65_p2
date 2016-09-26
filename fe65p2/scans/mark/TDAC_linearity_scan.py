
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
    "repeat_command": 100,
    "scan_range": [0.0, 0.2, 0.5],
    "vthin1Dac": 30,
    "vthin2Dac" :30,
    "PrmpVbpDac": 36,
    "preCompVbnDac" : 110,
    "columns" : [True] * 2 + [False] * 14,
    "clock_en1" : True,#True
    "clock_en2" : True,#True
    "trigger_en" : False,#False
    "offset" : 15,
    "pixels" : 512, #64 #512
    "measure_direction" : True, #True high to low #False low to high
    #"mask_filename": '/media/mark/1TB/Scanresults/output_data/new_chips/chip1_00/20160824_130427_noise_scan.h5'
}

class NoiseThreshScan(ScanBase):
    scan_id = "noise_scan_thresh"


    def scan(self, clock_en1=True, pixels=512, clock_en2=True, trigger_en=True, measure_direction=True,  offset=15, mask_steps=4, repeat_command=100, PrmpVbpDac=80, vthin2Dac=0, columns = [True] * 16, scan_range = [0, 0.2, 0.005], vthin1Dac = 80, preCompVbnDac = 50, mask_filename='', **kwargs):

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
        try:
            dut = Dut(ScanBase.get_basil_dir(self)+'/examples/lab_devices/agilent33250a_pyserial.yaml')
            dut.init()
            logging.info('Connected to '+str(dut['Pulser'].get_info()))
        except RuntimeError:
            INJ_LO = 0.2
            inj_factor = 2.0
            logging.info('External injector not connected. Switch to internal one')
            self.dut['INJ_LO'].set_voltage(INJ_LO, unit='V')

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
        self.dut['control']['PIX_D_CONF'] = 0#0
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
        #mask_tdac = np.full([64,64], 16, dtype = np.uint8)
        ###
        if pixels>1 and pixels<=64:
            mask_en[0:1, :] = True
        ###
        if pixels==1:
            mask_en[0][3]=True

        #for inx, col in enumerate(columns):
           #if col:
                #mask_en[inx*4:(inx+1)*4,:]  = True

        if mask_filename:
            logging.info('Using pixel mask from file: %s', mask_filename)

            with tb.open_file(mask_filename, 'r') as in_file_h5:
                #mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                if pixels>64:
                    mask_en = in_file_h5.root.scan_results.en_mask[:]

        mask_en = np.full([64, 64], True, dtype=np.bool)
        self.dut.write_en_mask(mask_en)
        #self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['OneSr'] = 0
        self.dut.write_global()

        self.dut['trigger'].set_delay(10000) #trigger for no injection
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
        m=80
        self.dut['global_conf']['vthin1Dac'] = 100
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['preCompVbnDac'] = 10
        #self.dut['global_conf']['PrmpVbpDac'] = 36
        self.dut.write_global()
        np.set_printoptions(threshold=np.inf)
        while (m==80):
            while(n==0):
                print "m: ", m, " n: ", n
                self.dut['global_conf']['vthin1Dac'] = 255
                self.dut['global_conf']['vthin2Dac'] = 0
                self.dut['global_conf']['preCompVbnDac'] = 50
                # self.dut['global_conf']['PrmpVbpDac'] = 36
                self.dut.write_global()
                for ni in range(1,32):
                    time.sleep(0.5)
                    self.dut['global_conf']['OneSr'] = 1  #all multi columns in parallel
                    self.dut.write_global()
                    mask_en = np.full([64, 64], False, dtype=np.bool)
                    self.dut.write_en_mask(mask_en)
                    self.dut.write_global()
                    with self.readout(scan_param_id = ni):
                        logging.info('Scan Parameter: %f (%d of %d)', ni, ni+1, 32)
                        mask_tdac = np.full([64, 64], ni, dtype=np.uint8)
                        print mask_tdac[0][0]
                        self.dut.write_tune_mask(mask_tdac)
                        self.dut['global_conf']['OneSr'] = 0  # all multi columns in parallel
                        self.dut.write_global()
                        mask_en = np.full([64, 64], True, dtype=np.bool)
                        self.dut.write_en_mask(mask_en)
                        self.dut.write_global()
                        time.sleep(0.1)
                n=n+10
                #self.dut['global_conf']['vthin1Dac'] = 255
                #self.dut['global_conf']['vthin2Dac'] = 255
                #self.dut['global_conf']['preCompVbnDac'] = 100
                #self.dut.write_global()
                #time.sleep(0.01)
            m=m+10
            n=0
        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Masks')
        self.h5_file.createCArray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.createCArray(scan_results, 'en_mask', obj=mask_en)

    def analyze(self,trigger, clock1, clock2, measuredirection,ranges):
        h5_filename = self.output_filename +'.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            ranges[1] = ranges[1] + 1
            ranges[0]= ranges[0] - 1
            attrs_tdc = {}
            with tb.open_file(h5_filename, 'r+') as in_file_h5:
                meta_data = in_file_h5.root.meta_data[:]
                en_mask = in_file_h5.root.scan_results.en_mask[:]
                new_meta = []
                if len(meta_data):
                    for elements in meta_data:
                        if elements[7] > 0:
                            if len(new_meta) < 1:
                                new_meta = elements
                            else:
                                new_meta = np.vstack((new_meta, elements))
                    param = []
                    noise = []
                    if len(new_meta):
                        param = np.arange(ranges[0], ranges[1])
                        for numbers in param:
                            noise_per_param = 0
                            for entries in new_meta:
                                if numbers == entries['scan_param_id']:
                                    noise_per_param = noise_per_param + entries['data_length']
                            noise = np.append(noise, noise_per_param/10)
                    en_mask1 = np.array(en_mask, dtype=bool)
                    pixels = np.count_nonzero(en_mask1)
                attrs_tdc['trigger']=trigger
                attrs_tdc['clock1']=clock1
                attrs_tdc['clock2'] = clock2
                attrs_tdc['direction']=measuredirection
                attrs_tdc['pixels']=pixels
                avg_tab = np.rec.fromarrays([param, noise], dtype=[('vthin1Dac', float), ('noise', int)])


                tdc_data=in_file_h5.createTable(in_file_h5.root, 'tdc_data', avg_tab, filters=self.filter_tables)
                tdc_data.attrs.clock_trigger = attrs_tdc
        plt.title("Trigger: " + str(trigger) + " Clock1: " + str(clock1) + " Clock2: " + str(clock2) + " high to low: " + str(measuredirection) + " Pixels: " + str(pixels))
        plt.plot(param, noise, linestyle='None', marker='o')
        plt.errorbar(param, noise, fmt='ro', yerr=np.sqrt(noise))
        #plt.ylim([0.1,10000000000])
        plt.xlim([np.min(param)-1,np.max(param)+1])
        plt.xlabel('global threshold Dac')
        plt.ylabel('noise [1/s]')
        plt.yscale('symlog')
        plt.draw()
        plt.savefig(h5_filename[:-3]+".png")
        plt.close()



if __name__ == "__main__":

    scan = NoiseThreshScan()
    scan.start(**local_configuration)
    #scan.analyze(local_configuration['trigger_en'], local_configuration['clock_en1'], local_configuration['clock_en2'],local_configuration['measure_direction'],[local_configuration['offset'],local_configuration['vthin1Dac']])
