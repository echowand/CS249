# assumes libffm is on PATH

PARAMS='-s 12 -k 5 -l 0.000001 -t 5'

cd ffm


# fold 0
echo "a"
ffm-train $PARAMS -p ffm_xgb_0_0.txt ffm_xgb_0_1.txt ffm_0_0.bin
echo "b"
ffm-predict ffm_xgb_0_0.txt ffm_0_0.bin pred_0_0.txt
echo "c"
ffm-train $PARAMS -p ffm_xgb_0_1.txt ffm_xgb_0_0.txt ffm_0_1.bin
echo "d"
ffm-predict ffm_xgb_0_1.txt ffm_0_1.bin pred_0_1.txt
# ffm-train $PARAMS ffm_xgb_0.txt ffm_0_full.bin


# fold 1
echo "a1"
ffm-train $PARAMS -p ffm_xgb_1_0.txt ffm_xgb_1_1.txt ffm_1_0.bin
ffm-predict ffm_xgb_1_0.txt ffm_1_0.bin pred_1_0.txt
echo "c1"
ffm-train $PARAMS -p ffm_xgb_1_1.txt ffm_xgb_1_0.txt ffm_1_1.bin
ffm-predict ffm_xgb_1_1.txt ffm_1_1.bin pred_1_1.txt

# ffm-train $PARAMS ffm_xgb_1.txt ffm_1_full.bin
echo "done"

# predict for test

# ffm-predict ffm_xgb_test_0.txt ffm_0_full.bin pred_test_0.txt
# ffm-predict ffm_xgb_test_1.txt ffm_1_full.bin pred_test_1.txt