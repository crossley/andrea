from util_funcs import *
from scipy import misc

# create s1 filter bank
filter_bank_s1 = make_s1_filter_bank()

# grab and save a test image
ascent = misc.ascent()

# get s1 output
window_size = 512 // 4
s1 = get_s1(ascent, filter_bank_s1, window_size)

# get c1 output
c1 = get_c1(s1)

# train s2 layer
filter_bank_s2 = train_s2(c1, filter_bank_s1, window_size)

# # get s2 output
# s2 = get_s2(c1)

# # get c2 output
# c2 = get_c2(s2)

# # inspect s1 output
# inspect_s1(s1)

# # inspect c1 output
# inspect_c1(c1)
