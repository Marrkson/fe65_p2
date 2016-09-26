
from fe65p2.scan_base import ScanBase
import fe65p2.plotting as  plotting
import time
import fe65p2.analysis as analysis


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
    "vthin1Dac": 20,
    "PrmpVbpDac": 80,
    "preCompVbnDac" : 110,
    "columns" : [True] * 2 + [False] * 14,
    "mask_filename": '/media/mark/1TB/Scanresults/output_data/new_chips/chip1/20160906_120820_noise_scan_standard.h5'
}

class ThresholdScan(ScanBase):
    scan_id = "threshold_scan"


    def scan(self, mask_steps=4, repeat_command=100, PrmpVbpDac=80, vthin2Dac=0, columns = [True] * 16, scan_range = [0, 0.2, 0.005], vthin1Dac = 80, preCompVbnDac = 50, mask_filename='', **kwargs):

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

        for inx, col in enumerate(columns):
           if col:
                mask_en[inx*4:(inx+1)*4,:]  = True

        if mask_filename:
            logging.info('Using pixel mask from file: %s', mask_filename)

            with tb.open_file(mask_filename, 'r') as in_file_h5:
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['OneSr'] = 0
        self.dut.write_global()

        self.dut['inj'].set_delay(10000) #this seems to be working OK problem is probably bad injection on GPAC usually +0
        self.dut['inj'].set_width(1000)
        self.dut['inj'].set_repeat(repeat_command)
        self.dut['inj'].set_en(False)

        self.dut['trigger'].set_delay(400-4)
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(1)
        self.dut['trigger'].set_en(True)

        ###
        self.dut['trigger'].set_delay(10000) #this seems to be working OK problem is probably bad injection on GPAC usually +0
        self.dut['trigger'].set_width(16)
        self.dut['trigger'].set_repeat(repeat_command)
        self.dut['trigger'].set_en(False)

        ##
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

        scan_range = np.arange(scan_range[0], scan_range[1], scan_range[2]) / inj_factor

        for idx, k in enumerate(scan_range):
            dut['Pulser'].set_voltage(INJ_LO, float(INJ_LO + k), unit='V')
            self.dut['INJ_HI'].set_voltage( float(INJ_LO + k), unit='V')
            time.sleep(0.5)

            bv_mask = bitarray.bitarray(lmask)

            with self.readout(scan_param_id = idx):
                logging.info('Scan Parameter: %f (%d of %d)', k, idx+1, len(scan_range))
                pbar = ProgressBar(maxval=mask_steps).start()
                for i in range(mask_steps):

                    self.dut['global_conf']['vthin1Dac'] = 255
                    self.dut['global_conf']['preCompVbnDac'] = 50
                    self.dut['global_conf']['vthin2Dac'] = 0
                    self.dut['global_conf']['PrmpVbpDac'] = 80
                    self.dut.write_global()
                    time.sleep(0.1)

                    self.dut['pixel_conf'][:]  = bv_mask
                    self.dut.write_pixel_col()
                    self.dut['global_conf']['InjEnLd'] = 1
                    #self.dut['global_conf']['PixConfLd'] = 0b11
                    self.dut.write_global()

                    bv_mask[1:] = bv_mask[0:-1]
                    bv_mask[0] = 0

                    self.dut['global_conf']['vthin1Dac'] = vthin1Dac
                    self.dut['global_conf']['preCompVbnDac'] = preCompVbnDac
                    self.dut['global_conf']['vthin2Dac'] = vthin2Dac
                    self.dut['global_conf']['PrmpVbpDac'] = PrmpVbpDac
                    self.dut.write_global()
                    time.sleep(0.1)

                    #self.dut['inj'].start()


                    pbar.update(i)

                    while not self.dut['inj'].is_done():
                        pass

                    while not self.dut['trigger'].is_done():
                        pass

                    #self.dut['trigger'].set_repeat(0)
                    #self.dut['control']['CLK_BX_GATE'] = 0
                    #self.dut['control']['CLK_OUT_GATE'] = 0
                    #self.dut['control'].write()

                    print('!!!!!!!!!!!!!!!1')
                    self.dut['tdc'].ENABLE = True
                    self.dut['trigger'].start()
                    time.sleep(10)
                    self.dut['tdc'].ENABLE = False
                    print('!!!!!!!!!!!!!!!2')

                    #self.dut['control']['CLK_BX_GATE'] = 1
                    #self.dut['control']['CLK_OUT_GATE'] = 1
                    #self.dut['control'].write()



        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Masks')
        self.h5_file.createCArray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.createCArray(scan_results, 'en_mask', obj=mask_en)


    def analyze(self):
        h5_filename = self.output_filename +'.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]

            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.createTable(in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)

        analysis.analyze_threshold_scan(h5_filename)
        status_plot = plotting.plot_status(h5_filename)
        occ_plot, H = plotting.plot_occupancy(h5_filename)
        tot_plot,_ = plotting.plot_tot_dist(h5_filename)
        lv1id_plot, _ = plotting.plot_lv1id_dist(h5_filename)
        scan_pix_hist, _ = plotting.scan_pix_hist(h5_filename)
        t_dac = plotting.t_dac_plot(h5_filename)

        output_file(self.output_filename + '.html', title=self.run_name)
        save(vplot(hplot(occ_plot, tot_plot, lv1id_plot), scan_pix_hist, t_dac, status_plot))

if __name__ == "__main__":

    scan = ThresholdScan()
    scan.start(**local_configuration)
    scan.analyze()
