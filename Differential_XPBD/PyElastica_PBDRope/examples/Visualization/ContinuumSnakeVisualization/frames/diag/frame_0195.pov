#include "../default.inc"

camera{
    location <15.0,10.5,-15.0>
    angle 30
    look_at <4.0,2.7,2.0>
    sky <0,1,0>
    right x*image_width/image_height
}
light_source{
    <15.0,10.5,-15.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<0.0011556533499821784,-0.0021905072421476383,4.8638019582131955>,0.05
    ,<-0.012037805863480175,-0.0017635533845997563,4.9120001793789925>,0.05
    ,<-0.030403295429956262,-0.001347276949277251,4.95847992993283>,0.05
    ,<-0.05608842850385784,-0.0009609544598376948,5.001359737377859>,0.05
    ,<-0.08868455566841106,-0.0006329795925168675,5.039264335223303>,0.05
    ,<-0.12639341855477912,-0.0003867045352331613,5.072101736022908>,0.05
    ,<-0.16707226770613431,-0.00022726014064110013,5.101193427298189>,0.05
    ,<-0.2086304522437058,-0.00014161821241383602,5.129023109981703>,0.05
    ,<-0.2486277095765693,-0.0001030953420597212,5.159051689682187>,0.05
    ,<-0.28331780632672077,-7.927938636287736e-05,5.195070528504322>,0.05
    ,<-0.3070248088998822,-4.353263901968141e-05,5.2390864642534325>,0.05
    ,<-0.31351904510515866,1.0552364004038483e-05,5.288641416668585>,0.05
    ,<-0.2999835313977563,5.9055866847380426e-05,5.3367406310526>,0.05
    ,<-0.2698202221956645,6.846292355382086e-05,5.376577713328608>,0.05
    ,<-0.23013091384310963,2.2188219725760873e-05,5.4069498803014575>,0.05
    ,<-0.18716358848928413,-7.635908521371127e-05,5.432491829364681>,0.05
    ,<-0.14525292158710762,-0.00020968702267154909,5.459737911540664>,0.05
    ,<-0.10815490766002059,-0.0003562549391265927,5.493239941083187>,0.05
    ,<-0.07881051741659197,-0.0004988890923042671,5.533703610044667>,0.05
    ,<-0.057454530335333556,-0.0006302951928035066,5.578895753486574>,0.05
    ,<-0.04097960453211524,-0.0007531467367607974,5.626088995381735>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
