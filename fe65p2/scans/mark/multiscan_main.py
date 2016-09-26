from fe65p2.scans.noise_scan import NoiseScan
import multiscan_plotting as plotting
import multiscan_base as scan_base
import os
import shutil
import time

def noise_scan_PrmpVbpDac(output_file, noisescan):
    #noisescan = NoiseScan()
    if not os.path.exists(output_file + '/PrmpVbpDac_' + time.strftime("%d%m%Y")):
        os.mkdir(output_file + '/PrmpVbpDac_' + time.strftime("%d%m%Y"))
    for n in range(0,8):
        namesaver=noisescan.output_filename
        a=namesaver.rfind("/")
        output_file=output_file+'/PrmpVbpDac_' + time.strftime("%d%m%Y")

        noisescan.output_filename = output_file + namesaver[a:] + '_PrmpVbpDac_' + str("%02d" % (n))

        if not os.path.exists(noisescan.output_filename):
            os.mkdir(noisescan.output_filename)

        print noisescan.output_filename[-38:]
        noisescan.output_filename = noisescan.output_filename + '/' + noisescan.output_filename[-38:]
        stepsize = 10
        for i in range(1, 10):
            print stepsize * i
            noisescan.output_filename = noisescan.output_filename + '_' + str("%03d" % (stepsize * i))
            noisescan.start(columns=([False] * 2 * n + [True] * 2 + [False] * (14 - 2 * n)), stop_pixel_count=4,
                            preCompVbnDac=50, PrmpVbpDac=stepsize * i)
            # noisescan.analyze()
            noisescan.output_filename = noisescan.output_filename[:-4]
        noisescan.output_filename=namesaver
    return output_file

def noise_scan_preCompVbnDac(output_file, noisescan):
    #noisescan = NoiseScan()
    if not os.path.exists(output_file + '/preCompVbnDac_' + time.strftime("%d%m%Y")):
        os.mkdir(output_file + '/preCompVbnDac_' + time.strftime("%d%m%Y"))
    for n in range(0,8):
        namesaver=noisescan.output_filename
        a=namesaver.rfind("/")
        output_file=output_file+'/preCompVbnDac_' + time.strftime("%d%m%Y")
        noisescan.output_filename = output_file + namesaver[a:] + '_preCompVbnDac_' + str("%02d" % (n))
        if not os.path.exists(noisescan.output_filename):
            os.mkdir(noisescan.output_filename)
        print noisescan.output_filename[-41:]
        noisescan.output_filename = noisescan.output_filename + '/' + noisescan.output_filename[-41:]
        stepsize = 10
        for i in range(1, 15):
            print stepsize * i
            noisescan.output_filename = noisescan.output_filename + '_' + str("%03d" % (stepsize * i))
            noisescan.start(columns=([False] * 2 * n + [True] * 2 + [False] * (14 - 2 * n)), stop_pixel_count=4,
                            preCompVbnDac=i*stepsize, PrmpVbpDac=36)
            # noisescan.analyze()
            noisescan.output_filename = noisescan.output_filename[:-4]
        noisescan.output_filename=namesaver
    return output_file

def noise_scan_vthin2Dac(output_file, noisescan):
    #noisescan = NoiseScan()
    if not os.path.exists(output_file + '/vthin2Dac_' + time.strftime("%d%m%Y")):
        os.mkdir(output_file + '/vthin2Dac_' + time.strftime("%d%m%Y"))
    for n in range(0,8):
        namesaver=noisescan.output_filename
        a=namesaver.rfind("/")
        output_file=output_file+'/vthin2Dac_' + time.strftime("%d%m%Y")
        noisescan.output_filename = output_file + namesaver[a:] + '_vthin2Dac_' + str("%02d" % (n))
        if not os.path.exists(noisescan.output_filename):
            os.mkdir(noisescan.output_filename)
        print noisescan.output_filename[-37:]
        noisescan.output_filename = noisescan.output_filename + '/' + noisescan.output_filename[-37:]
        stepsize = 20
        for i in range(0, 6):
            print stepsize * i
            noisescan.output_filename = noisescan.output_filename + '_' + str("%03d" % (stepsize * i))
            noisescan.start(columns=([False] * 2 * n + [True] * 2 + [False] * (14 - 2 * n)), vthin1Dac=i*stepsize+130, stop_pixel_count=4,
                            preCompVbnDac=50, vthin2Dac=i*stepsize, PrmpVbpDac=36)
            # noisescan.analyze()
            noisescan.output_filename = noisescan.output_filename[:-4]
        noisescan.output_filename=namesaver
    return output_file

if __name__ == "__main__":
    noisescan = NoiseScan()
    #noise_out1 = noise_scan_PrmpVbpDac('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/noisescans', noisescan)
    noise_out2 = noise_scan_preCompVbnDac('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/noisescans', noisescan)
    #noise_out3 = noise_scan_vthin2Dac('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/noisescans', noisescan)
    #out=scan_base.multirowthreshscan('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/noisescans/PrmpVbpDac_06092016', '/media/mark/1TB/Scanresults/output_data/new_chips/chip1/external')
    #scan_base.multirowtimescan('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/external/PrmpVbpDac_06092016')
    #plotting.full_plot(out,'vthin2Dac')
    pass