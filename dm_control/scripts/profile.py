import sys
import pstats

p = pstats.Stats('prof.out')
p.sort_stats('cumtime').print_stats(50)