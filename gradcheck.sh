#nohup ./SparseLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/Sparselabeler.out 2>&1 &
#nohup ./SparseCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseCRFMMlabeler.out 2>&1 &
#nohup ./SparseCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseCRFMLlabeler.out 2>&1 &
#nohup ./TNNLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/TNNlabeler.out 2>&1 &
#nohup ./TNNCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/TNNCRFMMlabeler.out 2>&1 &
#nohup ./TNNCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/TNNCRFMLlabeler.out 2>&1 &
#nohup ./SparseTNNLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseTNNlabeler.out 2>&1 &
#nohup ./SparseTNNCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseTNNCRFMMlabeler.out 2>&1 &
#nohup ./SparseTNNCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseTNNCRFMLlabeler.out 2>&1 &
#nohup ./RNNLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/RNNlabeler.out 2>&1 &
#nohup ./RNNCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/RNNCRFMMlabeler.out 2>&1 &
#nohup ./RNNCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/RNNCRFMLlabeler.out 2>&1 &
#nohup ./SparseRNNLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseRNNlabeler.out 2>&1 &
#nohup ./SparseRNNCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseRNNCRFMMlabeler.out 2>&1 &
#nohup ./SparseRNNCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseRNNCRFMLlabeler.out 2>&1 &

#nohup ./GatedLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/Gatedlabeler.out 2>&1 &
#nohup ./GatedCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/GatedCRFMMlabeler.out 2>&1 &
#nohup ./GatedCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/GatedCRFMLlabeler.out 2>&1 &

#nohup ./SparseGatedLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseGatedlabeler.out 2>&1 &
#nohup ./SparseGatedCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseGatedCRFMMlabeler.out 2>&1 &
#nohup ./SparseGatedCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseGatedCRFMLlabeler.out 2>&1 &

#nohup ./LSTMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/LSTMlabeler.out 2>&1 &
#nohup ./LSTMCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/LSTMCRFMMlabeler.out 2>&1 &
#nohup ./LSTMCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/LSTMCRFMLlabeler.out 2>&1 &

#nohup ./SparseLSTMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseLSTMlabeler.out 2>&1 &
#nohup ./SparseLSTMCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseLSTMCRFMMlabeler.out 2>&1 &
#nohup ./SparseLSTMCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseLSTMCRFMLlabeler.out 2>&1 &

nohup ./GRNNLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/GRNNlabeler.out 2>&1 &
nohup ./GRNNCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/GRNNCRFMMlabeler.out 2>&1 &
nohup ./GRNNCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/GRNNCRFMLlabeler.out 2>&1 &

nohup ./SparseGRNNLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseGRNNlabeler.out 2>&1 &
nohup ./SparseGRNNCRFMMLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseGRNNCRFMMlabeler.out 2>&1 &
nohup ./SparseGRNNCRFMLLabeler -l -train CNNER/train.feature -dev CNNER/dev.feature -test CNNER/test.feature -option CNNER/option.debug 1>debug/SparseGRNNCRFMLlabeler.out 2>&1 &
