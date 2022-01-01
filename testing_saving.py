import numpy

# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to csv file
savetxt('test_csv.csv', data, delimiter=',')

# save numpy array as npy file
from numpy import asarray
from numpy import save
# define data
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to npy file
save('test_data.npy', data)

#TESTING WITH TIMESTRING
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
string = 'weehee' + timestr + '_results.csv'
save('idabida' + timestr + '_results.csv', data)
save('weehee' + timestr + '_results.csv', data)