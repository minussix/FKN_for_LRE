# FKN_for_LRE
This is a official implementation for articla 'Koopman Theory Assisted Transfer Learning for Anomaly Detection of Liquid Rocket Engines in Frequency Domain'.

# LRSIMU Dataset
The simulation time seires classfication dataset we established can be found in [LRSIMU](https://drive.google.com/drive/folders/1ncBJxMuA2ovmPEyZ107btcix_vWU3CgW?usp=drive_link)

# Requirements
- Tensorflow
- Numpy
- Time
- Matplotlib
- Sklearn

# All Training Results on UCR Archives 2018
Data of UCR Archives 2018 can be found in [UCR Archives 2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
| Dataset                          | Precision   | Accuracy   | Recall     | Duration    |
|----------------------------------|-------------|------------|------------|-------------|
| ACSF1                            | 0.9633      | 0.9600     | 0.9600     | 1121.3832   |
| Adiac                            | 0.8483      | 0.8414     | 0.8410     | 1517.3554   |
| ArrowHead                        | 0.8227      | 0.8229     | 0.8225     | 1335.8578   |
| Beef                             | 0.8881      | 0.8667     | 0.8667     | 582.3503    |
| BeetleFly                        | 0.9545      | 0.9500     | 0.9500     | 605.2958    |
| BirdChicken                      | 0.8846      | 0.8500     | 0.8500     | 604.0760    |
| BME                              | 1.0000      | 1.0000     | 1.0000     | 1057.0743   |
| Car                              | 0.9201      | 0.9167     | 0.9174     | 621.5378    |
| CBF                              | 0.9945      | 0.9944     | 0.9944     | 4080.9533   |
| Chinatown                        | 0.9824      | 0.9883     | 0.9887     | 3134.3535   |
| ChlorineConcentration            | 0.7905      | 0.7964     | 0.7630     | 4187.2397   |
| CinCECGTorso                     | 0.8506      | 0.8493     | 0.8486     | 5793.3711   |
| Coffee                           | 1.0000      | 1.0000     | 1.0000     | 764.5825    |
| Computers                        | 0.8219      | 0.8080     | 0.8080     | 1554.9123   |
| CricketX                         | 0.8355      | 0.8256     | 0.8332     | 1654.3756   |
| CricketY                         | 0.7956      | 0.7897     | 0.7904     | 1641.8253   |
| CricketZ                         | 0.8060      | 0.8051     | 0.7979     | 1634.0036   |
| DiatomSizeReduction              | 0.3521      | 0.5882     | 0.4837     | 5051.0439   |
| DistalPhalanxOutlineAgeGroup     | 0.7665      | 0.7770     | 0.7503     | 1160.9305   |
| DistalPhalanxOutlineCorrect      | 0.8183      | 0.8116     | 0.7925     | 2043.9032   |
| DistalPhalanxTW                  | 0.4576      | 0.7050     | 0.5033     | 1175.4750   |
| Earthquakes                      | 0.7168      | 0.7626     | 0.5475     | 1465.2577   |
| ECG200                           | 0.9436      | 0.9500     | 0.9488     | 577.9282    |
| ECG5000                          | 0.6845      | 0.9433     | 0.5682     | 5110.5833   |
| ECGFiveDays                      | 1.0000      | 1.0000     | 1.0000     | 7005.3167   |
| ElectricDevices                  | 0.7263      | 0.7181     | 0.6653     | 32822.9440  |
| EOGHorizontalSignal              | 0.5744      | 0.5331     | 0.5338     | 3134.2361   |
| EOGVerticalSignal                | 0.4662      | 0.4033     | 0.4051     | 3139.5904   |
| EthanolLevel                     | 0.8088      | 0.8060     | 0.8062     | 5646.3629   |
| FaceAll                          | 0.8740      | 0.8639     | 0.9236     | 2960.3186   |
| FaceFour                         | 0.9741      | 0.9659     | 0.9677     | 1117.9589   |
| FacesUCR                         | 0.9425      | 0.9512     | 0.9246     | 2458.4341   |
| FiftyWords                       | 0.6204      | 0.7604     | 0.6088     | 1958.8410   |
| Fish                             | 0.9945      | 0.9943     | 0.9940     | 896.5306    |
| FreezerRegularTrain              | 0.9968      | 0.9968     | 0.9968     | 2850.5558   |
| FreezerSmallTrain                | 0.7705      | 0.7698     | 0.7698     | 17741.2252  |
| Fungi                            | 0.4416      | 0.6398     | 0.5462     | 3630.6704   |
| GunPoint                         | 0.9872      | 0.9867     | 0.9865     | 802.1725    |
| GunPointAgeSpan                  | 0.9968      | 0.9968     | 0.9969     | 887.5458    |
| GunPointMaleVersusFemale         | 1.0000      | 1.0000     | 1.0000     | 879.1725    |
| GunPointOldVersusYoung           | 1.0000      | 1.0000     | 1.0000     | 874.0761    |
| Ham                              | 0.7625      | 0.7619     | 0.7625     | 702.6678    |
| HandOutlines                     | 0.9500      | 0.9514     | 0.9439     | 14214.0265  |
| Haptics                          | 0.4902      | 0.4903     | 0.4877     | 1599.7479   |
| Herring                          | 0.7628      | 0.7188     | 0.6660     | 731.8977    |
| HouseTwenty                      | 0.9929      | 0.9916     | 0.9900     | 1249.0064   |
| InlineSkate                      | 0.4718      | 0.4745     | 0.4858     | 2580.4875   |
| InsectEPGRegularTrain            | 1.0000      | 1.0000     | 1.0000     | 1189.1302   |
| InsectEPGSmallTrain              | 0.5792      | 0.8313     | 0.6667     | 4584.6296   |
| InsectWingbeatSound              | 0.5997      | 0.6061     | 0.6061     | 2341.0784   |
| ItalyPowerDemand                 | 0.9699      | 0.9699     | 0.9699     | 3188.4877   |
| LargeKitchenAppliances           | 0.9173      | 0.9173     | 0.9173     | 2309.4444   |
| Lightning2                       | 0.8912      | 0.8852     | 0.8912     | 677.8987    |
| Lightning7                       | 0.7895      | 0.7945     | 0.8162     | 637.8394    |
| LRREAL                           | 1.0000      | 1.0000     | 1.0000     | 2111.9088   |
| Mallat                           | 0.9768      | 0.9761     | 0.9762     | 6568.5040   |
| Meat                             | 1.0000      | 1.0000     | 1.0000     | 609.6656    |
| MedicalImages                    | 0.7341      | 0.7579     | 0.7175     | 1760.7899   |
| MiddlePhalanxOutlineAgeGroup     | 0.6869      | 0.6494     | 0.5163     | 1224.1276   |
| MiddlePhalanxOutlineCorrect      | 0.8481      | 0.8282     | 0.8089     | 2018.3105   |
| MiddlePhalanxTW                  | 0.6236      | 0.6299     | 0.4370     | 1283.9340   |
| MixedShapesRegularTrain          | 0.9563      | 0.9571     | 0.9570     | 6842.0130   |
| MixedShapesSmallTrain            | 0.9005      | 0.9035     | 0.9009     | 5736.3998   |
| MoteStrain                       | 0.8460      | 0.8442     | 0.8406     | 8558.2326   |
| NonInvasiveFetalECGThorax1       | 0.9391      | 0.9410     | 0.9395     | 11225.8283  |
| OliveOil                         | 0.8785      | 0.8667     | 0.8264     | 592.3197    |
| OSULeaf                          | 0.9087      | 0.9050     | 0.8957     | 1040.3927   |
| Phoneme                          | 0.1584      | 0.2621     | 0.1357     | 4331.1588   |
| PigAirwayPressure                | 0.6989      | 0.6923     | 0.6923     | 1856.3246   |
| PigArtPressure                   | 1.0000      | 1.0000     | 1.0000     | 1851.1543   |
| PigCVP                           | 0.9683      | 0.9615     | 0.9615     | 1853.2905   |
| Plane                            | 1.0000      | 1.0000     | 1.0000     | 652.3713    |
| PowerCons                        | 1.0000      | 1.0000     | 1.0000     | 763.3010    |
| ProximalPhalanxOutlineAgeGroup   | 0.9230      | 0.8878     | 0.7253     | 1244.3844   |
| ProximalPhalanxOutlineCorrect    | 0.9305      | 0.9313     | 0.9088     | 2005.1580   |
| ProximalPhalanxTW                | 0.5271      | 0.8293     | 0.5351     | 1235.8302   |
| RefrigerationDevices             | 0.6076      | 0.6107     | 0.6107     | 2307.9347   |
| Rock                             | 0.7693      | 0.8200     | 0.7817     | 994.3366    |
| ScreenType                       | 0.6222      | 0.6133     | 0.6133     | 2305.2142   |
| SemgHandGenderCh2                | 0.8912      | 0.8700     | 0.8253     | 3593.2238   |
| SemgHandMovementCh2              | 0.5936      | 0.5467     | 0.5467     | 4451.7828   |
| SemgHandSubjectCh2               | 0.6884      | 0.6778     | 0.6778     | 4453.5805   |
| ShapeletSim                      | 1.0000      | 1.0000     | 1.0000     | 1672.4962   |
| ShapesAll                        | 0.8701      | 0.8600     | 0.8600     | 3123.2454   |
| SmallKitchenAppliances           | 0.8295      | 0.8293     | 0.8293     | 2301.2229   |
| SmoothSubspace                   | 1.0000      | 1.0000     | 1.0000     | 599.8132    |
| SonyAIBORobotSurface1            | 0.9578      | 0.9601     | 0.9617     | 4820.4153   |
| SonyAIBORobotSurface2            | 0.8812      | 0.8867     | 0.9019     | 7369.7863   |
| StarLightCurves                  | 0.9788      | 0.9791     | 0.9600     | 19803.1773  |
| Strawberry                       | 0.9772      | 0.9811     | 0.9819     | 2254.3180   |
| SwedishLeaf                      | 0.9582      | 0.9568     | 0.9574     | 1936.2766   |
| Symbols                          | 0.8658      | 0.8603     | 0.8640     | 7558.7900   |
| SyntheticControl                 | 0.6139      | 0.6200     | 0.6200     | 1092.2391   |
| ToeSegmentation1                 | 0.9779      | 0.9781     | 0.9792     | 1110.2572   |
| ToeSegmentation2                 | 0.9227      | 0.9385     | 0.8656     | 1114.2006   |
| Trace                            | 1.0000      | 1.0000     | 1.0000     | 613.1365    |
| TwoLeadECG                       | 0.9956      | 0.9956     | 0.9956     | 8954.1455   |
| TwoPatterns                      | 0.2739      | 0.2755     | 0.2707     | 5857.8518   |
| UMD                              | 0.7975      | 0.7847     | 0.7847     | 1113.0967   |
| UWaveGestureLibraryAll           | 0.7030      | 0.7069     | 0.7083     | 10377.7364  |
| UWaveGestureLibraryX             | 0.6256      | 0.6245     | 0.6247     | 6114.9704   |
| UWaveGestureLibraryY             | 0.6367      | 0.6415     | 0.6420     | 6028.6257   |
| UWaveGestureLibraryZ             | 0.6349      | 0.6387     | 0.6398     | 6130.1837   |
| Wafer                            | 0.9869      | 0.9963     | 0.9939     | 8365.3511   |
| Wine                             | 0.9500      | 0.9444     | 0.9444     | 750.8374    |
| WordSynonyms                     | 0.4757      | 0.6160     | 0.4618     | 1381.4999   |
| Worms                            | 0.7999      | 0.8052     | 0.7588     | 1159.5949   |
| WormsTwoClass                    | 0.8638      | 0.8442     | 0.8258     | 1170.6252   |
| Yoga                             | 0.8898      | 0.8893     | 0.8877     | 4025.3091   |
