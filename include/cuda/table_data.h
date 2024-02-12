#ifndef TABLE_DATA_H_
#define TABLE_DATA_H_


__device__ unsigned char m_inv_lut[128] ={
    128, 126, 124, 122, 120, 118, 117, 115, 113, 111, 109, 108, 106, 104, 103, 101, 100, 98, 96, 95, 93, 92, 90, 89, 88, 86, 85, 83, 82, 81, 79, 78, 77, 76, 74, 73, 72, 71, 69, 68, 67, 66, 65, 64, 63, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 44, 43, 42, 41, 40, 39, 38, 37, 37, 36, 35, 34, 33, 33, 32, 31, 30, 30, 29, 28, 27, 27, 26, 25, 24, 24, 23, 22, 22, 21, 20, 20, 19, 18, 18, 17, 16, 16, 15, 14, 14, 13, 13, 12, 11, 11, 10, 10, 9, 9, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};



__device__ short s_ilut_ab[512]={ 
    16256,16264,16273,16282,16292,16303,16314,16326,16256,16264,16273,16282,16292,16303,16314,16326,
    16339,16353,16367,16383,16391,16400,16410,16419,16339,16353,16367,16383,16391,16400,16410,16419,
    16430,16441,16453,16466,16479,16494,16509,16519,16430,16441,16453,16466,16479,16494,16509,16519,
    16527,16537,16547,16557,16568,16580,16593,16606,16527,16537,16547,16557,16568,16580,16593,16606,
    16620,16636,16646,16655,16664,16674,16684,16695,16620,16636,16646,16655,16664,16674,16684,16695,
    16707,16719,16733,16747,16762,16773,16782,16791,16707,16719,16733,16747,16762,16773,16782,16791,
    16801,16811,16822,16834,16846,16860,16874,16889,16801,16811,16822,16834,16846,16860,16874,16889,
    16900,16909,16918,16928,16938,16949,16961,16973,16900,16909,16918,16928,16938,16949,16961,16973,
    16986,17000,17015,17028,17036,17045,17055,17065,16986,17000,17015,17028,17036,17045,17055,17065,
    17076,17088,17100,17113,17127,17142,17155,17163,17076,17088,17100,17113,17127,17142,17155,17163,
    17172,17182,17192,17203,17215,17227,17240,17254,17172,17182,17192,17203,17215,17227,17240,17254,
    17269,17282,17291,17300,17309,17319,17330,17341,17269,17282,17291,17300,17309,17319,17330,17341,
    17354,17367,17381,17395,17410,17418,17427,17436,17354,17367,17381,17395,17410,17418,17427,17436,
    17446,17457,17468,17481,17494,17507,17522,17537,17446,17457,17468,17481,17494,17507,17522,17537,
    17545,17554,17563,17573,17584,17595,17607,17620,17545,17554,17563,17573,17584,17595,17607,17620,
    17634,17649,17664,17672,17681,17690,17700,17711,17634,17649,17664,17672,17681,17690,17700,17711,
    14768,14779,14791,14804,14818,14832,14848,14856,14768,14779,14791,14804,14818,14832,14848,14856,
    14865,14874,14884,14895,14906,14918,14931,14945,14865,14874,14884,14895,14906,14918,14931,14945,
    14959,14974,14983,14992,15001,15011,15022,15033,14959,14974,14983,14992,15001,15011,15022,15033,
    15045,15058,15071,15086,15101,15111,15119,15129,15045,15058,15071,15086,15101,15111,15119,15129,
    15138,15149,15160,15172,15185,15198,15212,15228,15138,15149,15160,15172,15185,15198,15212,15228,
    15238,15247,15256,15266,15276,15287,15299,15311,15238,15247,15256,15266,15276,15287,15299,15311,
    15325,15339,15354,15365,15374,15383,15393,15403,15325,15339,15354,15365,15374,15383,15393,15403,
    15414,15426,15438,15452,15466,15481,15492,15501,15414,15426,15438,15452,15466,15481,15492,15501,
    15510,15520,15530,15541,15553,15565,15578,15592,15510,15520,15530,15541,15553,15565,15578,15592,
    15607,15620,15628,15637,15647,15657,15668,15680,15607,15620,15628,15637,15647,15657,15668,15680,
    15692,15705,15719,15734,15747,15755,15764,15774,15692,15705,15719,15734,15747,15755,15764,15774,
    15784,15795,15806,15819,15832,15846,15861,15874,15784,15795,15806,15819,15832,15846,15861,15874,
    15883,15892,15901,15911,15922,15933,15946,15959,15883,15892,15901,15911,15922,15933,15946,15959,
    15972,15987,16001,16010,16019,16028,16038,16049,15972,15987,16001,16010,16019,16028,16038,16049,
    16060,16073,16085,16099,16114,16129,16137,16146,16060,16073,16085,16099,16114,16129,16137,16146,
    16155,16165,16176,16187,16199,16212,16226,16240,16155,16165,16176,16187,16199,16212,16226,16240
};

__device__ short s_ilut_cd[512]={ 
    16256,16199,16155,16113,16060,16018,15972,15921,16256,16199,16155,16113,16060,16018,15972,15921,
    15882,15831,15784,15746,15691,15646,15607,15552,15882,15831,15784,15746,15691,15646,15607,15552,
    15510,15465,15414,15373,15324,15275,15237,15184,15510,15465,15414,15373,15324,15275,15237,15184,
    15138,15101,15045,15001,14959,14906,14864,14817,15138,15101,15045,15001,14959,14906,14864,14817,
    14767,14728,14677,14630,14593,14537,14492,14452,14767,14728,14677,14630,14593,14537,14492,14452,
    14398,14356,14310,14259,14220,14170,14121,14084,14398,14356,14310,14259,14220,14170,14121,14084,
    14030,13984,13946,13890,13847,13804,13752,13711,14030,13984,13946,13890,13847,13804,13752,13711,
    13663,13613,13575,13522,13476,13439,13383,13339,13663,13613,13575,13522,13476,13439,13383,13339,
    13297,13244,13202,13156,13105,13066,13015,12967,13297,13244,13202,13156,13105,13066,13015,12967,
    12930,12875,12830,12791,12736,12693,12649,12597,12930,12875,12830,12791,12736,12693,12649,12597,
    12557,12508,12459,12421,12368,12322,12284,12228,12557,12508,12459,12421,12368,12322,12284,12228,
    12185,12142,12090,12048,12001,11951,11912,11861,12185,12142,12090,12048,12001,11951,11912,11861,
    11814,11777,11721,11676,11636,11582,11540,11494,11814,11777,11721,11676,11636,11582,11540,11494,
    11443,11403,11354,11305,11268,11214,11168,11129,11443,11403,11354,11305,11268,11214,11168,11129,
    11074,11031,10988,10935,10895,10847,10797,10759,11074,11031,10988,10935,10895,10847,10797,10759,
    10706,10660,10623,10567,10523,10481,10428,10386,10706,10660,10623,10567,10523,10481,10428,10386,
    10340,10289,10250,10199,10151,10114,10059,10014,10340,10289,10250,10199,10151,10114,10059,10014,
    9975,9920,9877,9833,9781,9741,9692,9643,9975,9920,9877,9833,9781,9741,9692,9643,
    9605,9552,9506,9468,9412,9369,9326,9273,9605,9552,9506,9468,9412,9369,9326,9273,
    9232,9185,9135,9096,9045,8997,8961,8905,9232,9185,9135,9096,9045,8997,8961,8905,
    8860,8820,8766,8724,8678,8627,8587,8537,8860,8820,8766,8724,8678,8627,8587,8537,
    8489,8452,8397,8352,8313,8258,8215,8171,8489,8452,8397,8352,8313,8258,8215,8171,
    8119,8079,8030,7981,7943,7890,7843,7807,8119,8079,8030,7981,7943,7890,7843,7807,
    7750,7706,7665,7611,7570,7523,7473,7434,7750,7706,7665,7611,7570,7523,7473,7434,
    7383,7335,7298,7243,7198,7158,7104,7061,7383,7335,7298,7243,7198,7158,7104,7061,
    7017,6965,6925,6876,6827,6789,6736,6690,7017,6965,6925,6876,6827,6789,6736,6690,
    6652,6596,6553,6510,6457,6416,6369,6319,6652,6596,6553,6510,6457,6416,6369,6319,
    6280,6228,6181,6145,6089,6044,6003,5949,6280,6228,6181,6145,6089,6044,6003,5949,
    5907,5862,5811,5771,5721,5673,5636,5581,5907,5862,5811,5771,5721,5673,5636,5581,
    5536,5497,5442,5399,5355,5303,5262,5214,5536,5497,5442,5399,5355,5303,5262,5214,
    5165,5127,5074,5027,4991,4934,4890,4849,5165,5127,5074,5027,4991,4934,4890,4849,
    4795,4754,4707,4657,4618,4567,4519,4482,4795,4754,4707,4657,4618,4567,4519,4482
};

__device__ short s_flut_ab[512]={ 
    16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,
    16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,16256,
    16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,
    16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,
    16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,
    16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,16257,
    16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,
    16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,
    16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,
    16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,16258,
    16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,
    16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,
    16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,16259,
    16259,16259,16259,16259,16259,16259,16259,16260,16259,16259,16259,16259,16259,16259,16259,16260,
    16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,
    16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,
    16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,16260,
    16260,16260,16260,16260,16260,16260,16261,16261,16260,16260,16260,16260,16260,16260,16261,16261,
    16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,
    16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,
    16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,16261,
    16261,16261,16261,16261,16261,16262,16262,16262,16261,16261,16261,16261,16261,16262,16262,16262,
    16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,
    16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,
    16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,16262,
    16262,16262,16262,16263,16263,16263,16263,16263,16262,16262,16262,16263,16263,16263,16263,16263,
    16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,
    16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,
    16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,16263,
    16263,16263,16264,16264,16264,16264,16264,16264,16263,16263,16264,16264,16264,16264,16264,16264,
    16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,
    16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264,16264
};

__device__ short s_flut_cd[512]={ 
    16256,16255,16255,16255,16255,16254,16254,16254,16256,16255,16255,16255,16255,16254,16254,16254,
    16254,16253,16253,16253,16253,16252,16252,16252,16254,16253,16253,16253,16253,16252,16252,16252,
    16252,16251,16251,16251,16251,16250,16250,16250,16252,16251,16251,16251,16251,16250,16250,16250,
    16250,16249,16249,16249,16249,16248,16248,16248,16250,16249,16249,16249,16249,16248,16248,16248,
    16248,16247,16247,16247,16247,16246,16246,16246,16248,16247,16247,16247,16247,16246,16246,16246,
    16246,16245,16245,16245,16245,16244,16244,16244,16246,16245,16245,16245,16245,16244,16244,16244,
    16244,16244,16243,16243,16243,16243,16242,16242,16244,16244,16243,16243,16243,16243,16242,16242,
    16242,16242,16241,16241,16241,16241,16240,16240,16242,16242,16241,16241,16241,16241,16240,16240,
    16240,16240,16240,16239,16239,16239,16239,16238,16240,16240,16240,16239,16239,16239,16239,16238,
    16238,16238,16238,16237,16237,16237,16237,16236,16238,16238,16238,16237,16237,16237,16237,16236,
    16236,16236,16236,16236,16235,16235,16235,16235,16236,16236,16236,16236,16235,16235,16235,16235,
    16234,16234,16234,16234,16234,16233,16233,16233,16234,16234,16234,16234,16234,16233,16233,16233,
    16233,16232,16232,16232,16232,16231,16231,16231,16233,16232,16232,16232,16232,16231,16231,16231,
    16231,16231,16230,16230,16230,16230,16229,16229,16231,16231,16230,16230,16230,16230,16229,16229,
    16229,16229,16229,16228,16228,16228,16228,16227,16229,16229,16229,16228,16228,16228,16228,16227,
    16227,16227,16227,16227,16226,16226,16226,16226,16227,16227,16227,16227,16226,16226,16226,16226,
    16225,16225,16225,16225,16225,16224,16224,16224,16225,16225,16225,16225,16225,16224,16224,16224,
    16224,16223,16223,16223,16223,16223,16222,16222,16224,16223,16223,16223,16223,16223,16222,16222,
    16222,16222,16221,16221,16221,16221,16221,16220,16222,16222,16221,16221,16221,16221,16221,16220,
    16220,16220,16220,16220,16219,16219,16219,16219,16220,16220,16220,16220,16219,16219,16219,16219,
    16218,16218,16218,16218,16218,16217,16217,16217,16218,16218,16218,16218,16218,16217,16217,16217,
    16217,16217,16216,16216,16216,16216,16215,16215,16217,16217,16216,16216,16216,16216,16215,16215,
    16215,16215,16215,16214,16214,16214,16214,16214,16215,16215,16215,16214,16214,16214,16214,16214,
    16213,16213,16213,16213,16213,16212,16212,16212,16213,16213,16213,16213,16213,16212,16212,16212,
    16212,16212,16211,16211,16211,16211,16210,16210,16212,16212,16211,16211,16211,16211,16210,16210,
    16210,16210,16210,16209,16209,16209,16209,16209,16210,16210,16210,16209,16209,16209,16209,16209,
    16208,16208,16208,16208,16208,16207,16207,16207,16208,16208,16208,16208,16208,16207,16207,16207,
    16207,16207,16206,16206,16206,16206,16206,16205,16207,16207,16206,16206,16206,16206,16206,16205,
    16205,16205,16205,16205,16204,16204,16204,16204,16205,16205,16205,16205,16204,16204,16204,16204,
    16204,16203,16203,16203,16203,16203,16202,16202,16204,16203,16203,16203,16203,16203,16202,16202,
    16202,16202,16202,16201,16201,16201,16201,16201,16202,16202,16202,16201,16201,16201,16201,16201,
    16200,16200,16200,16200,16200,16199,16199,16199,16200,16200,16200,16200,16200,16199,16199,16199
};

#endif
