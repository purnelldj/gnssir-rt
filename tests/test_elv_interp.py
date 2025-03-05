from pytest import approx

from gnssir.elv_interp import elv_interp_array


# taking 1hr from one antenna
def test_elv_interp_array(snr_in, snr_elv_interp_out):
    snr_out = elv_interp_array(snr_in, kspac=1800)
    assert snr_in.shape[1] == snr_out.shape[1], "mismatching shapes from elv interp"
    assert approx(snr_elv_interp_out) == snr_out, "mismatching values from elv interp"
