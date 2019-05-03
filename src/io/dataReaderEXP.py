import dataReader

# create an instance
dr = dataReader.dataReader()

## STEP_ONE: 		data type settings:
# read 's1' data
dr.changeDatOpt('s1')
# read 's2' data
dr.changeDatOpt('s2')
# read 's1' and 's2' data
dr.changeDatOpt('both')


## STEP_TWO: 		data cross validation setting, in reproductive manner
# 1. the 1-5 folds randomly load a percentage (dr.trPerc) of data from cultural-10, put them into training; the rest as testing. 
# 2. the seednull option load all cultural-10 data as testing data.
# the 1 fold
dr.changeTrOpt('seed1')
# the 2 fold
dr.changeTrOpt('seed2')
# the 3 fold
dr.changeTrOpt('seed3')
# the 4 fold
dr.changeTrOpt('seed4')
# the 5 fold
dr.changeTrOpt('seed5')
# no fold
dr.changeTrOpt('seednull')

## STEP_THREE: 		data directory setting
dr.changeLocDir('/data/hu/TF/data')

## STEP_FOUR: 		data random picking percentage setting
dr.changePerc(.5)

## STEP_FIVE: 		call function 'loadLCZ' to load data
dat_tr,lab_tr,dat_te,lab_te = dr.loadLCZ()

## EXTRA THINGS:
# 1. load small data set for code debugging
# 	a. load one .h5 file 
dat,lab = dr.loadOneFile('/data/hu/TF/data/testCites/munich.h5')
#	b. train and test spliting
dat_tr,lab_tr,dat_te,lab_te = dr.randomSplit(dat,lab)




