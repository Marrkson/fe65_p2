from fe65p2.scan_base import ScanBase
from fe65p2.fe65p2 import fe65p2
import time
import matplotlib.pyplot as plt
from bokeh.charts import HeatMap, bins, output_file, vplot, hplot, show
from bokeh.palettes import RdYlGn6, RdYlGn9, BuPu9, Spectral11
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, show, vplot, hplot, save
from progressbar import ProgressBar
import yaml
import os

local_configuration = {
    "stop_pixel_count": 64,
    "vthin2Dac" : 0,
    "vthin1Dac" : 130,
    "columns": [True] * 16, #+ [False] * 14,
    "preCompVbnDac": 115,
    "mask_filename": '/media/mark/1TB/Scanresults/output_data/new_chips/chip1/20160906_120820_noise_scan_standard.h5'
}

class NoiseWalk(ScanBase):
    scan_id = "noisewalk"

    def scan(self, columns = [True] * 16, PrmpVbpDac=80, stop_pixel_count = 4, preCompVbnDac = 110,  vthin2Dac = 0, vthin1Dac = 130, mask_filename='', **kwargs):

        if mask_filename:
            logging.info('Using pixel mask from file: %s', mask_filename)

            with tb.open_file(mask_filename, 'r') as in_file_h5:
                dac_status = yaml.load(in_file_h5.root.meta_data.attrs.dac_status)
                mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                mask_en = in_file_h5.root.scan_results.en_mask[:]
            vthin1Dac = dac_status['vthin1Dac']+10
            vthin2Dac = dac_status['vthin2Dac']
            preCompVbnDac = dac_status['preCompVbnDac']
            stop_pixel_count = 4
            PrmpVbpDac = dac_status['PrmpVbpDac']
        INJ_LO = 0.2
        self.dut['INJ_LO'].set_voltage(INJ_LO, unit='V')
        self.dut['INJ_HI'].set_voltage(INJ_LO, unit='V')

        self.dut['global_conf']['PrmpVbpDac'] = 80
        self.dut['global_conf']['vthin1Dac'] = 255
        self.dut['global_conf']['vthin2Dac'] = 0
        self.dut['global_conf']['vffDac'] = 24
        self.dut['global_conf']['PrmpVbnFolDac'] = 51
        self.dut['global_conf']['vbnLccDac'] = 1
        self.dut['global_conf']['compVbnDac'] = 25
        self.dut['global_conf']['preCompVbnDac'] = 110  # 50

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

        self.dut['global_conf']['ColEn'][:] = bitarray.bitarray([True] * 16)  # (columns)
        self.dut['global_conf']['ColSrEn'][:] = bitarray.bitarray([True] * 16)
        self.dut.write_global()

        self.dut['pixel_conf'].setall(False)
        self.dut.write_pixel()
        self.dut['global_conf']['InjEnLd'] = 1
        self.dut.write_global()
        self.dut['global_conf']['InjEnLd'] = 0

        #mask_en = np.full([64, 64], True, dtype=np.bool)
        #mask_tdac = np.full([64, 64], 16, dtype=np.uint8)

        if mask_filename:
            logging.info('Using pixel mask from file: %s', mask_filename)

            #with tb.open_file(mask_filename, 'r') as in_file_h5:
                #mask_tdac = in_file_h5.root.scan_results.tdac_mask[:]
                #mask_en = in_file_h5.root.scan_results.en_mask[:]

        self.dut.write_en_mask(mask_en)
        self.dut.write_tune_mask(mask_tdac)

        self.dut['global_conf']['OneSr'] = 1
        self.dut.write_global()

        #exit()

        self.dut['trigger'].set_delay(100) #this seems to be working OK problem is probably bad injection on GPAC
        self.dut['trigger'].set_width(1) #try single
        self.dut['trigger'].set_repeat(100000)
        self.dut['trigger'].set_en(False)#False

        np.set_printoptions(linewidth=150)
        iteration = 0
        all_hits=()
        all_vthin1Dac=()
        for i in range(vthin1Dac):
            with self.readout(scan_param_id=vthin1Dac-i,fill_buffer = True, clear_buffer = True):
                logging.info('Scan Parameter: %f (%d of %d)', vthin1Dac -  i,i, vthin1Dac)
                logging.info('Scan iteration: %d (vthin1Dac = %d)', iteration, vthin1Dac)
                self.dut['global_conf']['vthin1Dac'] = 255
                self.dut['global_conf']['vthin2Dac'] = 0
                self.dut['global_conf']['preCompVbnDac'] = 110  # 50
                self.dut['global_conf']['PrmpVbpDac'] = 80
                self.dut.write_global()
                time.sleep(0.1)

                self.dut['tdc']['ENABLE'] = True
                self.dut['global_conf']['vthin1Dac'] = vthin1Dac-i
                self.dut['global_conf']['vthin2Dac'] = vthin2Dac
                self.dut['global_conf']['preCompVbnDac'] = preCompVbnDac
                self.dut['global_conf']['PrmpVbpDac'] = PrmpVbpDac
                self.dut.write_global()
                time.sleep(0.1)

                self.dut['trigger'].start()
                while not self.dut['trigger'].is_done():
                    pass

                dqdata = self.fifo_readout.data
                data = np.concatenate([item[0] for item in dqdata])
                hit_data = self.dut.interpret_raw_data(data)
                all_hits=np.append(all_hits,hit_data.shape[0])
                all_vthin1Dac=np.append(all_vthin1Dac,vthin1Dac-i)
                self.dut['tdc'].ENABLE = 0

            self.dut['global_conf']['vthin1Dac'] = 255
            self.dut['global_conf']['vthin2Dac'] = 0
            self.dut['global_conf']['preCompVbnDac'] = 110  # 50
            self.dut['global_conf']['PrmpVbpDac'] = 80
            self.dut.write_global()
            iteration += 1

        self.dut['global_conf']['vthin1Dac'] = vthin1Dac
        self.dut['global_conf']['vthin2Dac'] = vthin2Dac
        self.dut['global_conf']['preCompVbnDac'] = preCompVbnDac
        self.dut['global_conf']['PrmpVbpDac'] = PrmpVbpDac

        errors = np.sqrt(all_hits)
        avg_tab = np.rec.fromarrays([all_vthin1Dac, all_hits, errors],
                                    dtype=[('vthin1Dac', int), ('Hits', int),('Hit_error',float)])

        self.h5_file.createTable(self.h5_file.root, 'Hits', avg_tab, filters=self.filter_tables)

        scan_results = self.h5_file.create_group("/", 'scan_results', 'Scan Results')
        self.h5_file.createCArray(scan_results, 'tdac_mask', obj=mask_tdac)
        self.h5_file.createCArray(scan_results, 'en_mask', obj=mask_en)

    def interpret_noisewalk(self):
        h5_filename = self.output_filename +'.h5'
        with tb.open_file(h5_filename, 'r+') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]
            meta_data = in_file_h5.root.meta_data[:]
            hit_data = self.dut.interpret_raw_data(raw_data, meta_data)
            in_file_h5.createTable(in_file_h5.root, 'hit_data', hit_data, filters=self.filter_tables)
            hits = in_file_h5.root.Hits[:]
        all_vthin1Dac=()
        all_hits=()
        error=()
        for i in hits:
            all_vthin1Dac=np.append(all_vthin1Dac,i[0])
            all_hits = np.append(all_hits, i[1])
            error=np.append(error,i[2])
        plt.xlabel("vthin1Dac")
        plt.ylabel("Noise Hits")
        plt.plot(all_vthin1Dac, all_hits, 'bo')
        plt.errorbar(all_vthin1Dac, all_hits, fmt='bo', yerr=error)
        plt.savefig(self.output_filename+'_1.png')
        plt.close()
        #plt.show()


    def plot_occupancy(self):
        h5_file_name = self.output_filename + '.h5'
        with tb.open_file(h5_file_name, 'r') as in_file_h5:
            hit_data = in_file_h5.root.hit_data[:]
            heatmap_array=np.zeros((64,64))
        for i in hit_data:
            heatmap_array[i[2],i[1]]=heatmap_array[i[2],i[1]]+1
        x = np.arange(0, 64, 1)
        y = np.arange(0, 64, 1)
        plt.pcolormesh(x, y, heatmap_array)
        plt.xlabel("column")
        plt.ylabel("row")
        plt.colorbar()  # need a colorbar to show the intensity scale
        plt.savefig(self.output_filename+'_2.png')
        plt.close()


if __name__ == "__main__":

    scan = NoiseWalk()
    scan.start(**local_configuration)
    scan.interpret_noisewalk()
    scan.plot_occupancy()