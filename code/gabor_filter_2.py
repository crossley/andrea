from util_funcs import *
from scipy import misc

# create s1 filter bank
filter_bank = make_s1_filter_bank()

# grab and save a test image
ascent = misc.ascent()

# get s1 output
window_size = 512 // 4
s1 = get_s1(ascent, filter_bank, window_size)

# inspect s1 output
inspect_s1(s1)

# get c1 output
c1 = get_c1(s1)

# inspect c1 output
inspect_c1(c1)
