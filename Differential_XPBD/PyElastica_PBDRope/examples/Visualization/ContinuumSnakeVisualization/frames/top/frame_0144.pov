#include "../default.inc"

camera{
    location <0,15,3>
    angle 30
    look_at <0.0,0,3>
    sky <-1,0,0>
    right x*image_width/image_height
}
light_source{
    <0.0,8.0,5.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<-0.2172991646192497,-0.002214535421273615,3.4411195385568476>,0.05
    ,<-0.2291761398808855,-0.001764835160599547,3.4896630575168857>,0.05
    ,<-0.23531284257287632,-0.0013157616735365212,3.539263833618947>,0.05
    ,<-0.2317851437225887,-0.0008800615930909339,3.5891222798053763>,0.05
    ,<-0.2171326938951866,-0.00048754992845793,3.6369148607621895>,0.05
    ,<-0.19221635078205415,-0.00017808638967714566,3.68025806009481>,0.05
    ,<-0.15935648412517472,1.7756391285423724e-05,3.7179472599747463>,0.05
    ,<-0.12139209483185605,9.246277946585007e-05,3.750499916629808>,0.05
    ,<-0.08123098229448507,6.163475274857539e-05,3.7803075818874214>,0.05
    ,<-0.04212218745005785,-4.1142040207467665e-05,3.8114826688735173>,0.05
    ,<-0.008545870342528595,-0.00017260536183477296,3.848542374332975>,0.05
    ,<0.013122215035774255,-0.000290435113185478,3.8936001329041736>,0.05
    ,<0.016449940558263054,-0.0003666236822267103,3.9434738489549344>,0.05
    ,<-0.0005410778408795725,-0.0003961897275935146,3.9904728376953966>,0.05
    ,<-0.03329873510512915,-0.00039510239826290317,4.028214422362406>,0.05
    ,<-0.07445854179554633,-0.0003868547567740406,4.056566883298835>,0.05
    ,<-0.11830970133975117,-0.000385331471988536,4.0805598177542395>,0.05
    ,<-0.16122726599349146,-0.0003900193652982846,4.1061914023032875>,0.05
    ,<-0.2004868649488379,-0.00039291962543414716,4.137137876219697>,0.05
    ,<-0.23469161036585381,-0.0003875976214062992,4.173595194845524>,0.05
    ,<-0.26510326878437906,-0.0003749774189610822,4.213274069517662>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
