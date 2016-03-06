mkdir -p output/c3d/v_ApplyEyeMakeup_g01_c01
mkdir -p output/c3d/v_BaseballPitch_g01_c01
cd input/frm
tar xvzf v_ApplyEyeMakeup_g01_c01.tar.gz
tar xvzf v_BaseballPitch_g01_c01.tar.gz
cd ../..
GLOG_logtosterr=1 /imatge/amontes/src/caffe-c3d/build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_frm.prototxt /imatge/amontes/src/caffe-c3d/examples/c3d_feature_extraction/conv3d_deepnetA_sport1m_iter_1900000 0 50 1 prototxt/output_list_prefix.txt fc7-1 fc6-1 prob
