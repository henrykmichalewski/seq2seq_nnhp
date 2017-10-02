# seq2seq_nnhp
Sequence two sequence applied to the Mizar dataset in the prefix notation

Training done on 4 Titan-X GPUs. The original data is [here](https://github.com/JUrban/deepmath/tree/master/nnhpdata). The code is the tensorflow implementation of [seq2seq with attention](https://github.com/tensorflow/nmt) (tensorflow 1.4 required). The hyperparemeters file, the dictionary (identical for premises and theorems) as well as the parallel corpus of theorems and premises are included in this repo. 

Examples of behavior after 10000 steps:

Example 1. Input:
```
* ! / b0  * * c=> * c~ * cv1_xboole_0 b0 * ! / b1  * * c=> * * c& * cv1_funct_1 b1 * * c& * * * cv1_funct_2 b1 b0 ck6_margrel1 * * cm1_subset_1 b1 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1 * ! / b2  * * c=> * * cm1_subset_1 b2 * ck1_zfmisc_1 * ck1_bvfunc_2 b0 * ! / b3  * * c=> * * cm1_eqrel_1 b3 b0 * ! / b4  * * c=> * * cm1_eqrel_1 b4 b0 * * c=> * * cv2_bvfunc_2 b2 b0 * * * cr1_bvfunc_1 b0 * * ck1_bvfunc_1 b0 * * * * ck7_bvfunc_2 b0 * * * * ck7_bvfunc_2 b0 b1 b2 b3 b2 b4 * * ck1_bvfunc_1 b0 * * * * ck7_bvfunc_2 b0 * * * * ck6_bvfunc_2 b0 b1 b2 b4 b2 b3  . 
```

Reference output:
```
* ! / b0  * ! / b1  * ! / b2  * ! / b3  * * c=> * * c& * c~ * cv1_xboole_0 b0 * * c& * * c& * cv1_funct_1 b1 * * c& * * * cv1_funct_2 b1 b0 ck6_margrel1 * * cm1_subset_1 b1 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1 * * c& * * cm1_subset_1 b2 * ck1_zfmisc_1 * ck1_bvfunc_2 b0 * * cm1_eqrel_1 b3 b0 * * c& * cv1_funct_1 * * * * ck6_bvfunc_2 b0 b1 b2 b3 * * c& * * * cv1_funct_2 * * * * ck6_bvfunc_2 b0 b1 b2 b3 b0 ck6_margrel1 * * cm1_subset_1 * * * * ck6_bvfunc_2 b0 b1 b2 b3 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1  . * ! / b0  * ! / b1  * ! / b2  * ! / b3  * * c=> * * c& * c~ * cv1_xboole_0 b0 * * c& * * c& * cv1_funct_1 b1 * * c& * * * cv1_funct_2 b1 b0 ck6_margrel1 * * cm1_subset_1 b1 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1 * * c& * * cm1_subset_1 b2 * ck1_zfmisc_1 * ck1_bvfunc_2 b0 * * cm1_eqrel_1 b3 b0 * * c& * cv1_funct_1 * * * * ck7_bvfunc_2 b0 b1 b2 b3 * * c& * * * cv1_funct_2 * * * * ck7_bvfunc_2 b0 b1 b2 b3 b0 ck6_margrel1 * * cm1_subset_1 * * * * ck7_bvfunc_2 b0 b1 b2 b3 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1  . * ! / b0  * * c=> * c~ * cv1_xboole_0 b0 * ! / b1  * * c=> * * c& * cv1_funct_1 b1 * * c& * * * cv1_funct_2 b1 b0 ck6_margrel1 * * cm1_subset_1 b1 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1 * ! / b2  * * c=> * * c& * cv1_funct_1 b2 * * c& * * * cv1_funct_2 b2 b0 ck6_margrel1 * * cm1_subset_1 b2 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1 * * c=> * * * cr1_bvfunc_1 b0 b1 b2 * * * cr1_bvfunc_1 b0 * * ck1_bvfunc_1 b0 b2 * * ck1_bvfunc_1 b0 b1  . * ! / b0  * * c=> * c~ * cv1_xboole_0 b0 * ! / b1  * * c=> * * c& * cv1_funct_1 b1 * * c& * * * cv1_funct_2 b1 b0 ck6_margrel1 * * cm1_subset_1 b1 * ck1_zfmisc_1 * * ck2_zfmisc_1 b0 ck6_margrel1 * ! / b2  * * c=> * * cm1_subset_1 b2 * ck1_zfmisc_1 * ck1_bvfunc_2 b0 * ! / b3  * * c=> * * cm1_eqrel_1 b3 b0 * ! / b4  * * c=> * * cm1_eqrel_1 b4 b0 * * c=> * * cv2_bvfunc_2 b2 b0 * * * cr1_bvfunc_1 b0 * * * * ck7_bvfunc_2 b0 * * * * ck6_bvfunc_2 b0 b1 b2 b3 b2 b4 * * * * ck7_bvfunc_2 b0 * * * * ck7_bvfunc_2 b0 b1 b2 b4 b2 b3  . 
```

Actual output:
```* ! / b0 * ! / b1 * ! / b2 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3 * ! / b3
```

Example 2. Input: 
```
* * c= ck7_monoid_0 * * cg3_algstr_0 ck5_numbers ck48_binop_2  . 
```

Reference output:
```
* ! / b0  * * c=> * cl3_algstr_0 b0 * * c=> * cv15_algstr_0 b0 * * c= b0 * * cg3_algstr_0 * cu1_struct_0 b0 * cu2_algstr_0 b0  . * ! / b0  * * c=> * * c& * c~ * cv2_struct_0 b0 * * c& * cv15_algstr_0 b0 * * c& * cv1_group_1 b0 * * c& * cv17_monoid_0 b0 * * cm2_monoid_0 b0 ck6_monoid_0 * * c<=> * * c= b0 ck7_monoid_0 * * c= * cu1_struct_0 b0 ck5_numbers  . * * c= ck48_binop_2 * cu2_algstr_0 ck7_monoid_0  . * * c& * c~ * cv2_struct_0 ck6_monoid_0 * * c& * cv15_algstr_0 ck6_monoid_0 * * c& * cv1_group_1 ck6_monoid_0 * * c& * cv3_group_1 ck6_monoid_0 * * c& * cv5_group_1 ck6_monoid_0 * cl3_algstr_0 ck6_monoid_0  . * * c& * c~ * cv2_struct_0 ck7_monoid_0 * * c& * cv15_algstr_0 ck7_monoid_0 * * c& * cv1_group_1 ck7_monoid_0 * * c& * cv17_monoid_0 ck7_monoid_0 * * cm2_monoid_0 ck7_monoid_0 ck6_monoid_0  . * ! / b0  * * c=> * cl3_algstr_0 b0 * ! / b1  * * c=> * * cm2_monoid_0 b1 b0 * cl3_algstr_0 b1  . 
```

Actual output:
```
* ! / b0 * * c=> * cv7_ordinal1 b0 * ! / b1 * * c=> * * cm1_subset_1 b1 * ck1_zfmisc_1 b0 * ! / b2 * * c=> * * cm1_subset_1 b2 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * cm1_subset_1 b3 * cu1_struct_0 b0 * ! / b3 * * c=> * * c& * c~ * cv2_struct_0 b3 *
```
