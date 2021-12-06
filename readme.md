# 8to1mult_hdf5.py
read binary data for drs4 digitizer in chunks using 'read_in_chunks' function.
open 8to1_mult.hdf5 to write the data (using d1) in a compressed format that is read from 8to1mult_999 (1).dat

d3 used to write time_bins.
d2 used to write time_cells or trigger cell.

files ends on line 223.


# 8to1mult_read_hdf5.py
read 8to1_mult.hdf5 data file.
'get_data' fn used to read hdf file
time_samples used as an array to store time sequences for each trigger cell from time_bins

'get_imp_response' fn to get impulse response (also see drs_fdm_parser class)
'recover_resonator_pulses' for pulse recovery
'get_amp_timing' get get pulse amplitude and timing
After line 351, we divide the pulse amplitude into energy ranges to get means and std for the errors in amp and timing. Done.